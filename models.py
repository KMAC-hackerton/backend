# app/models.py
import numpy as np
import torch
import torch.nn as nn
from config import (
    VesselSpec, CostWeights, CostParams, EnvFields, 
    EF_CO2, EF_BC, MIX_PHYS_AND_DL, KAPPA_UNCERT, device
)

# --- 유틸리티 함수 ---
def softplus(x, beta=1.0): 
    return (1.0/beta)*np.log1p(np.exp(beta*x))
def sigmoid(x): 
    return 1.0/(1.0+np.exp(-x))

# --- 물리 비용 모델 ---
class PhysicalCost:
    def __init__(self, vessel, w: CostWeights, p: CostParams):
        self.vessel, self.w, self.p = vessel, w, p
    def fuel_rate(self, v_kn, Hs, SIC, U10):
        v_term = v_kn**3
        M_sea = 1.0 + self.p.a_s*softplus(Hs - self.p.H0)
        M_ice = 1.0 + self.p.a_i*sigmoid(self.p.beta1*(SIC - self.p.s1))
        M_wind = 1.0 + 0.03*U10
        return self.vessel.alpha_fuel * v_term * M_sea * M_ice * M_wind
    def edge_cost(self, d_nm, v_kn, F: EnvFields, t_idx, y, x):
        ef_co2 = EF_CO2.get(self.vessel.fuel_type, 3.206)
        ef_bc = EF_BC.get(self.vessel.fuel_type, 2.5e-3)
        SIC, Hs, U10 = F.SIC[t_idx,y,x], F.Hs[t_idx,y,x], F.U10[t_idx,y,x]
        dist_ice, bio = F.dist_ice[t_idx,y,x], F.bio_mask[t_idx,y,x]
        grad, forbid = F.risk_grad_ice[t_idx,y,x], F.forbid_mask[t_idx,y,x]
        dt_h = d_nm / max(v_kn, 1e-6)
        mf = self.fuel_rate(v_kn, Hs, SIC, U10) * dt_h
        co2e = mf * ef_co2
        W_ice = 1/(1+np.exp(-self.p.gamma1*(SIC-0.15))) + np.exp(-self.p.gamma2*dist_ice)
        bc = mf * ef_bc * W_ice
        SL = self.p.a0 + self.p.a2*(v_kn**self.p.p)
        noise = SL * bio * dt_h
        risk = 0.5*(1/(1+np.exp(-self.p.kappa1*(SIC-self.p.s_c)))) + 0.3*softplus(Hs-self.p.Hs_c) + 0.2*softplus(grad-self.p.g_c)
        def nz(v): return v / (np.abs(v)+1e-9)
        c = self.w.w_fuel*nz(co2e) + self.w.w_bc*nz(bc) + self.w.w_noise*nz(noise) + self.w.w_risk*nz(risk)
        if forbid > 0.5: c += self.w.bigM
        return float(c)

class TinyNCF(nn.Module):
    def __init__(self, in_dim, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class NeuralCostFieldDL:
    def __init__(self, model, phys=None, mix=MIX_PHYS_AND_DL, use_mc=True, kappa=KAPPA_UNCERT):
        self.model, self.phys, self.mix = model.eval(), phys, float(mix)
        self.use_mc, self.kappa = use_mc, float(kappa)
        if self.use_mc:
            def enable_dropout(m):
                if isinstance(m, nn.Dropout): m.train()
            self.model.apply(enable_dropout)
    
    @torch.no_grad()
    def dl_cost(self, d_nm, v_kn, F, t, y, x, mc=10):
        T,Y,X = F.SIC.shape
        feats = np.array([[F.SIC[t,y,x], F.Hs[t,y,x], F.U10[t,y,x], F.dist_ice[t,y,x],
                           F.depth[t,y,x], F.bio_mask[t,y,x], F.risk_grad_ice[t,y,x], v_kn,
                           t/(T-1), y/(Y-1), x/(X-1)]], np.float32)
        xt = torch.from_numpy(feats).to(device)
        
        if not self.use_mc:
            return float(self.model(xt).item())
            
        preds = [self.model(xt).item() for _ in range(mc)]
        mu, sd = float(np.mean(preds)), float(np.std(preds))
        return mu + self.kappa * sd
        
    def predict(self, d_nm, v_kn, F, t, y, x):
        dlc = self.dl_cost(d_nm, v_kn, F, t, y, x)
        if self.phys is None or self.mix <= 0.0: return dlc
        pc = self.phys.edge_cost(d_nm, v_kn, F, t, y, x)
        return (1-self.mix)*dlc + self.mix*pc