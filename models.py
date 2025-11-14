from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import config


def softplus(x: float, beta: float = 1.0) -> float:
    return (1.0 / beta) * np.log1p(np.exp(beta * x))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class EnvFields:
    SIC: np.ndarray
    Hs: np.ndarray
    U10: np.ndarray
    dist_ice: np.ndarray
    depth: np.ndarray
    bio_mask: np.ndarray
    risk_grad_ice: np.ndarray
    forbid_mask: np.ndarray


@dataclass
class VesselSpec:
    name: str = "Generic"
    fuel_type: str = "MGO"
    alpha_fuel: float = 1.0
    v_min: float = 8.0
    v_max: float = 16.0
    bcf: float = 1.0


EF_CO2 = {"MGO": 3.206, "LSFO": 3.114, "LNG": 2.75, "eMeOH": 0.25}
EF_BC = {"MGO": 2.5e-3, "LSFO": 3.5e-3, "LNG": 0.7e-3, "eMeOH": 0.3e-3}


@dataclass
class CostWeights:
    w_fuel: float = 0.6
    w_bc: float = 0.25
    w_risk: float = 0.15
    bigM: float = 1e6


@dataclass
class CostParams:
    H0: float = 2.0
    a_s: float = 0.6
    s1: float = 0.15
    beta1: float = 8.0
    a_i: float = 1.0
    gamma1: float = 8.0
    gamma2: float = 0.02
    s_c: float = 0.35
    kappa1: float = 10.0
    Hs_c: float = 4.0
    g_c: float = 0.55
    a0: float = 150.0
    a2: float = 0.02
    p: float = 3.2


class PhysicalCost:
    def __init__(self, vessel: VesselSpec, w: CostWeights, p: CostParams):
        self.vessel = vessel
        self.w = w
        self.p = p

    def fuel_rate(self, v_kn: float, Hs: float, SIC: float, U10: float) -> float:
        v_term = v_kn ** 3
        M_sea = 1.0 + self.p.a_s * softplus(Hs - self.p.H0)
        M_ice = 1.0 + self.vessel.bcf * self.p.a_i * sigmoid(self.p.beta1 * (SIC - self.p.s1))
        M_wind = 1.0 + 0.03 * U10
        return self.vessel.alpha_fuel * v_term * M_sea * M_ice * M_wind

    def get_ice_class_limit(self) -> float:
        if self.vessel.bcf <= 0.5:
            return 0.90
        if self.vessel.bcf <= 1.0:
            return 0.70
        return 0.15

    def edge_cost(self, d_nm: float, v_kn: float, F: EnvFields, t_idx: int, y: int, x: int) -> float:
        ef_co2 = EF_CO2.get(self.vessel.fuel_type, 3.206)
        ef_bc = EF_BC.get(self.vessel.fuel_type, 2.5e-3)
        SIC = float(F.SIC[t_idx, y, x])
        Hs = float(F.Hs[t_idx, y, x])
        U10 = float(F.U10[t_idx, y, x])
        dist_ice = float(F.dist_ice[t_idx, y, x])
        bio = float(F.bio_mask[t_idx, y, x])
        grad = float(F.risk_grad_ice[t_idx, y, x])
        forbid = float(F.forbid_mask[t_idx, y, x])

        dt_h = d_nm / max(v_kn, 1e-6)
        mf = self.fuel_rate(v_kn, Hs, SIC, U10) * dt_h
        co2e = mf * ef_co2
        W_ice = 1 / (1 + np.exp(-self.p.gamma1 * (SIC - 0.15))) + np.exp(-self.p.gamma2 * dist_ice)
        bc = mf * ef_bc * W_ice
        SL = self.p.a0 + self.p.a2 * (v_kn ** self.p.p) if hasattr(self.p, 'a0') else self.p.H0
        noise = SL * bio * dt_h
        risk = (
            0.5 * (1 / (1 + np.exp(-self.p.kappa1 * (SIC - self.p.s_c))))
            + 0.3 * softplus(Hs - self.p.Hs_c)
            + 0.2 * softplus(grad - self.p.g_c)
        )
        def nz(v: float) -> float:
            return v / (np.abs(v) + 1e-9)

        c = self.w.w_fuel * nz(co2e) + self.w.w_bc * nz(bc) + self.w.w_risk * nz(risk)

        sic_max_limit = self.get_ice_class_limit()
        forbid_mask_dynamic = 1.0 if SIC > sic_max_limit else 0.0

        if forbid > 0.5 or forbid_mask_dynamic > 0.5:
            c += self.w.bigM

        return float(c)


class TinyNCF(nn.Module):
    def __init__(self, in_dim: int = 12, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralCostFieldDL:
    def __init__(
        self,
        model: TinyNCF,
        phys: Optional[PhysicalCost] = None,
        mix: float = config.MIX_PHYS_AND_DL,
        use_mc: bool = True,
        kappa: float = config.KAPPA_UNCERT,
    ):
        self.model = model.eval()
        self.phys = phys
        self.mix = float(mix)
        self.use_mc = use_mc
        self.kappa = float(kappa)
        if self.use_mc:
            def enable_dropout(module):
                if isinstance(module, nn.Dropout):
                    module.train()
            self.model.apply(enable_dropout)

    @torch.no_grad()
    def dl_cost(self, d_nm: float, v_kn: float, F: EnvFields, t: int, y: int, x: int, mc: int = 10) -> float:
        T, Y, X = F.SIC.shape
        if self.phys is None:
            bcf = 1.0
        else:
            bcf = self.phys.vessel.bcf
        feats = np.array([
            [
                F.SIC[t, y, x],
                F.Hs[t, y, x],
                F.U10[t, y, x],
                F.dist_ice[t, y, x],
                F.depth[t, y, x],
                F.bio_mask[t, y, x],
                F.risk_grad_ice[t, y, x],
                v_kn,
                t / max(T - 1, 1),
                y / max(Y - 1, 1),
                x / max(X - 1, 1),
                bcf,
            ]
        ], dtype=np.float32)
        xt = torch.from_numpy(feats).to(config.DEVICE)
        if not self.use_mc:
            return float(self.model(xt).item())
        preds = [self.model(xt).item() for _ in range(mc)]
        mu = float(np.mean(preds))
        sd = float(np.std(preds))
        return mu + self.kappa * sd

    def predict(self, d_nm: float, v_kn: float, F: EnvFields, t: int, y: int, x: int) -> float:
        cost = self.dl_cost(d_nm, v_kn, F, t, y, x)
        if self.phys is None or self.mix <= 0.0:
            return cost
        phys_cost_val = self.phys.edge_cost(d_nm, v_kn, F, t, y, x)
        return (1 - self.mix) * cost + self.mix * phys_cost_val
