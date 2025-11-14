# app/utils.py
import math
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    EnvFields, VSET, ENV_DATA_PATH, EF_CO2, EF_BC, 
    CostParams, VesselSpec, CostWeights,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX # v5.0의 극지방 시각화용
)
from models import softplus, PhysicalCost

try:
    import cartopy.crs as ccrs
    import cartopy.feature
except ImportError:
    print("🚨 'cartopy' 모듈을 찾을 수 없습니다. 시각화는 2D 격자 맵으로 대체됩니다.")
    ccrs = None

# ----------------- (중요) 처리된 데이터 로더 -----------------
def load_processed_env_fields() -> EnvFields:
    """
    Colab/scripts가 생성한 .npz 파일에서 환경 데이터를 로드합니다.
    """
    try:
        data = np.load(ENV_DATA_PATH)
        return EnvFields(
            SIC=data['SIC'], Hs=data['Hs'], U10=data['U10'],
            dist_ice=data['dist_ice'], depth=data['depth'],
            bio_mask=data['bio_mask'], risk_grad_ice=data['risk_grad_ice'],
            forbid_mask=data['forbid_mask']
        )
    except FileNotFoundError:
        raise RuntimeError(f"Processed env data not found at {ENV_DATA_PATH}. "
                           "Colab에서 데이터를 전처리하고 outputs/ 폴더에 저장했는지 확인하세요.")
    except Exception as e:
        raise RuntimeError(f"Error loading processed env data: {e}")

# --- A* 경로 탐색 함수 ---
def astar_route_with_speeds(F, cost_model, start, goal, vset, d_cell_nm=2.7, max_iterations=50000):
    # (v3.4의 astar_route_with_speeds 함수 코드)
    T,Y,X = F.SIC.shape
    (t0,y0,x0), (tG,yG,xG) = start, goal
    def inb(t,y,x): return 0<=t<T and 0<=y<Y and 0<=x<X
    def h(t,y,x): return math.hypot(y-yG, x-xG)*d_cell_nm*1.0
    NEIGH = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
    pq = []; heapq.heappush(pq, (h(t0,y0,x0), 0.0, (t0,y0,x0)))
    g_cost, parent, best_speed_to = {(t0,y0,x0):0.0}, {}, {}
    iteration_count = 0
    while pq:
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"⚠️ A* 조기 종료: {max_iterations}회 반복 초과 (목표 미도달)")
            return None, None, float("inf")
        if iteration_count % 5000 == 0:
            print(f"[A*] 진행: {iteration_count}회, 큐={len(pq)}, 방문={len(g_cost)}")
        f, gc, node = heapq.heappop(pq)
        t,y,x = node
        if node == (tG,yG,xG):
            path=[node]; speeds=[]
            while node in parent:
                speeds.append(best_speed_to[node]); node=parent[node]; path.append(node)
            return list(reversed(path)), list(reversed(speeds)), gc
        nt = t+1
        for dy,dx in NEIGH:
            ny,nx = y+dy, x+dx
            if not inb(nt,ny,nx): continue
            best_c, best_v = float("inf"), None
            for v in vset:
                c = cost_model.predict(2.7, float(v), F, t, ny, nx)
                if c < best_c: best_c, best_v = c, v
            ng = gc + best_c; nnode = (nt,ny,nx)
            if ng < g_cost.get(nnode, float("inf")):
                g_cost[nnode]=ng; parent[nnode]=(t,y,x); best_speed_to[nnode]=best_v
                heapq.heappush(pq, (ng + h(nt,ny,nx), ng, nnode))
    return None, None, float("inf")

# --- 리포팅/시각화 함수 ---
def compute_cost_map(F, cost_model, t, vset=VSET):
    # (v3.4의 compute_cost_map 함수 코드)
    T,Y,X = F.SIC.shape
    assert 0 <= t < T
    cost_map = np.full((Y,X), np.nan, dtype=np.float32)
    for y in range(Y):
        for x in range(X):
            if F.forbid_mask[t,y,x] > 0.5:
                continue
            best = float("inf")
            for v in vset:
                c = cost_model.predict(2.7, float(v), F, t, y, x)
                if c < best: best = c
            cost_map[y,x] = best
    return cost_map

def make_latlon_grids(Y, X, lat_min, lat_max, lon_min, lon_max):
    # (v5.0의 헬퍼 함수)
    lats = np.linspace(lat_max, lat_min, Y) # Y가 북쪽(lat_max)에서 시작
    lons = np.linspace(lon_min, lon_max, X)
    return lats, lons

def plot_route_visualization(F, cost_model, path, savepath, vset=VSET):
    t_for_heat = path[0][0] if path else 0
    C = compute_cost_map(F, cost_model, t_for_heat, vset=vset)
    C[F.forbid_mask[t_for_heat] > 0.5] = np.nan
    vmin = np.nanpercentile(C, 5); vmax = np.nanpercentile(C, 95)
    
    Y, X = C.shape
    title = f"Optimized Arctic Route (Day {t_for_heat})"

    if ccrs:
        fig = plt.figure(figsize=(9, 9))
        stereo_proj = ccrs.NorthPolarStereo(central_longitude=0)
        ax = fig.add_subplot(1, 1, 1, projection=stereo_proj)
        data_crs = ccrs.PlateCarree()
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], data_crs)
        ax.set_title(title)
        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
        ax.add_feature(cartopy.feature.LAND, facecolor='lightgray')
        ax.add_feature(cartopy.feature.OCEAN, facecolor='skyblue')
        
        lats, lons = make_latlon_grids(Y, X, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        lon_2d, lat_2d = np.meshgrid(lons, lats)
        
        im = ax.pcolormesh(lon_2d, lat_2d, C, transform=data_crs,
                           cmap='magma', vmin=vmin, vmax=vmax, alpha=0.7, shading='auto')
        
        lats_of_path = np.array([lats[y] for _, y, _ in path])
        lons_of_path = np.array([lons[x] for _, _, x in path])
        
        ax.plot(lons_of_path, lats_of_path, color='cyan', linewidth=3, 
                marker='o', markersize=4, transform=data_crs, label="Optimized Route")
        
        fig.colorbar(im, ax=ax, orientation='vertical', label='Edge cost (a.u.)', pad=0.05)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Cartopy 없을 시 (v3.4)
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(C, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
        cb = plt.colorbar(im, ax=ax); cb.set_label("Edge cost (a.u.)")
        if path:
            ys = [y for (_,y,_) in path]; xs = [x for (_,_,x) in path]
            ax.plot(xs, ys, color="red", lw=2.5)

    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def edge_cost_breakdown(phys: PhysicalCost, F: EnvFields, t, y, x, v_kn, d_nm=2.7):
    ef_co2 = EF_CO2.get(phys.vessel.fuel_type, 3.206)
    ef_bc = EF_BC.get(phys.vessel.fuel_type, 2.5e-3)
    P = phys.p
    SIC, Hs, U10 = F.SIC[t,y,x], F.Hs[t,y,x], F.U10[t,y,x]
    dist_ice, bio = F.dist_ice[t,y,x], F.bio_mask[t,y,x]
    grad, forbid = F.risk_grad_ice[t,y,x], F.forbid_mask[t,y,x]
    dt_h = d_nm / max(v_kn, 1e-6)
    mf = phys.fuel_rate(v_kn, Hs, SIC, U10) * dt_h
    co2e = mf * ef_co2
    W_ice = 1/(1+np.exp(-P.gamma1*(SIC-0.15))) + np.exp(-P.gamma2*dist_ice)
    bc = mf * ef_bc * W_ice
    SL = P.a0 + P.a2*(v_kn**P.p)
    noise = SL * bio * dt_h
    risk = 0.5*(1/(1+np.exp(-P.kappa1*(SIC-P.s_c)))) + 0.3*softplus(Hs-P.Hs_c) + 0.2*softplus(grad-P.g_c)
    breakdown = {
        "CO2e": phys.w.w_fuel * co2e,
        "BC": phys.w.w_bc * bc,
        "Noise": phys.w.w_noise * noise,
        "Risk": phys.w.w_risk * risk,
        "Forbidden": phys.w.bigM if forbid > 0.5 else 0.0
    }
    return breakdown

def summarize_path_costs(phys: PhysicalCost, F: EnvFields, path, speeds):
    if not path or not speeds:
        return pd.DataFrame()
    totals = {"CO2e":0.0, "BC":0.0, "Noise":0.0, "Risk":0.0, "Forbidden":0.0}
    for (t0,y1,x1),(t1,y2,x2),v in zip(path[:-1], path[1:], speeds):
        br = edge_cost_breakdown(phys, F, t0, y2, x2, float(v))
        for k in totals: totals[k] += br[k]
    df = pd.DataFrame([
        ("CO₂e (proxy)", totals["CO2e"], "a.u."),
        ("Black Carbon (proxy)", totals["BC"], "a.u."),
        ("Underwater Noise (proxy)", totals["Noise"], "a.u."),
        ("Risk (unitless)", totals["Risk"], "a.u."),
        ("Forbidden Zone Hits", totals["Forbidden"] / phys.w.bigM, "count")
    ], columns=["Metric","Value","Unit"])
    return df