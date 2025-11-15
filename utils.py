import heapq
import math
from typing import Tuple, List, Optional, Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from models import EnvFields, PhysicalCost, NeuralCostFieldDL, CostWeights

try:
    import cartopy.crs as ccrs  # type: ignore[import]
    import cartopy.feature as cfeature  # type: ignore[import]
except ImportError:
    ccrs = None
    cfeature = None

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxesSubplot  # type: ignore[import]
else:
    GeoAxesSubplot = Any


def load_processed_env_fields() -> EnvFields:
    try:
        data = np.load(config.ENV_DATA_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Environment data not found at {config.ENV_DATA_PATH}") from exc
    return EnvFields(
        SIC=data['SIC'],
        Hs=data['Hs'],
        U10=data['U10'],
        dist_ice=data['dist_ice'],
        depth=data['depth'],
        bio_mask=data['bio_mask'],
        risk_grad_ice=data['risk_grad_ice'],
        forbid_mask=data['forbid_mask'],
    )


def astar_route_with_speeds(
    F: EnvFields,
    cost_model: NeuralCostFieldDL,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    vset: Tuple[float, ...] = config.VSET,
    d_cell_nm: float = 2.7,
    max_expansions: int = 200000,
) -> Tuple[Optional[List[Tuple[int, int, int]]], Optional[List[float]], float]:
    """
    시간 축은 항상 t+1 로만 진행.
    - goal의 시간 인덱스를 tG로 두고, t >= tG 인 노드는 더 이상 확장하지 않음.
    - 탐색 노드 개수가 max_expansions를 넘으면 강제 중단하고 (None, None, inf) 리턴.
    """
    T, Y, X = F.SIC.shape
    (t0, y0, x0), (tG, yG, xG) = start, goal

    def inb(t, y, x):
        return 0 <= t < T and 0 <= y < Y and 0 <= x < X

    def h(t, y, x):
        # 공간 거리 기반 휴리스틱
        return math.hypot(y - yG, x - xG) * d_cell_nm * 1.0

    NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    pq = []
    heapq.heappush(pq, (h(t0, y0, x0), 0.0, (t0, y0, x0)))
    g_cost = {(t0, y0, x0): 0.0}
    parent = {}
    best_speed_to = {}

    expansions = 0

    while pq:
        f, gc, node = heapq.heappop(pq)
        t, y, x = node

        # 이미 더 좋은 경로가 등록되어 있으면 스킵
        if gc > g_cost.get(node, float("inf")):
            continue

        # 목표 도달
        if node == (tG, yG, xG):
            path = [node]
            speeds = []
            while node in parent:
                speeds.append(best_speed_to[node])
                node = parent[node]
                path.append(node)
            return list(reversed(path)), list(reversed(speeds)), gc

        # 목표 시간 tG를 넘은 노드는 확장하지 않음
        if t >= tG:
            continue

        expansions += 1
        if expansions > max_expansions:
            # 탐색 과도 → 실패 처리
            break

        nt = t + 1
        if nt > tG:
            # 목표 시간보다 더 나중 시간은 볼 필요 없음
            continue

        for dy, dx in NEIGH:
            ny, nx = y + dy, x + dx
            if not inb(nt, ny, nx):
                continue

            # 하드 포비든 셀은 아예 확장 안 함
            if F.forbid_mask[nt, ny, nx] > 0.5:
                continue

            best_c, best_v = float("inf"), None
            for v in vset:
                c = cost_model.predict(d_cell_nm, float(v), F, t, ny, nx)
                # Big-M 수준의 비용은 사실상 막힌 셀로 간주하고 스킵
                if c >= CostWeights().bigM * 0.5:
                    continue
                if c < best_c:
                    best_c, best_v = c, v

            # 모든 속도가 Big-M 수준이면 이 이웃은 봉인
            if best_v is None:
                continue

            ng = gc + best_c
            nnode = (nt, ny, nx)

            if ng < g_cost.get(nnode, float("inf")):
                g_cost[nnode] = ng
                parent[nnode] = (t, y, x)
                best_speed_to[nnode] = best_v
                heapq.heappush(pq, (ng + h(nt, ny, nx), ng, nnode))

    # 여기까지 왔다는 건 목표까지 경로를 찾지 못했다는 뜻
    return None, None, float("inf")


def compute_cost_map(F: EnvFields, cost_model: NeuralCostFieldDL, t: int, vset: Tuple[float, ...] = config.VSET) -> np.ndarray:
    T, Y, X = F.SIC.shape
    assert 0 <= t < T
    cost_map = np.full((Y, X), np.nan, dtype=np.float32)
    for y in range(Y):
        for x in range(X):
            if F.forbid_mask[t, y, x] > 0.5:
                continue
            best = float('inf')
            for v in vset:
                c = cost_model.predict(2.7, float(v), F, t, y, x)
                if c < best:
                    best = c
            cost_map[y, x] = best
    return cost_map


def make_latlon_grids(Y: int, X: int, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(lat_min, lat_max, Y, dtype=np.float32)
    lons = np.linspace(lon_min, lon_max, X, dtype=np.float32)
    return lats, lons


def plot_route_visualization(
    F: EnvFields,
    cost_model: NeuralCostFieldDL,
    path: List[Tuple[int, int, int]],
    savepath: str,
    vset: Tuple[float, ...] = config.VSET,
):
    if not path:
        return
    t_for_heat = path[0][0]
    C = compute_cost_map(F, cost_model, t_for_heat, vset)
    C[F.forbid_mask[t_for_heat] > 0.5] = np.nan
    vmin = float(np.nanpercentile(C, 5))
    vmax = float(np.nanpercentile(C, 95))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    if path:
        ys = [y for _, y, _ in path]
        xs = [x for _, _, x in path]
        ax.plot(xs, ys, color="cyan", lw=2.5)
    ax.set_title(f"Optimized Arctic Route (Day {t_for_heat})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Edge cost (a.u.)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_route_on_polar_stereo(
    F: EnvFields,
    cost_model: NeuralCostFieldDL,
    path: List[Tuple[int, int, int]],
    savepath: str,
    *,
    lat_min: float = config.LAT_MIN,
    lat_max: float = config.LAT_MAX,
    lon_min: float = config.LON_MIN,
    lon_max: float = config.LON_MAX,
    vset: Tuple[float, ...] = config.VSET,
):
    if ccrs is None or cfeature is None:
        print("[polar] Cartopy not available; skipping polar visualization")
        return
    if not path:
        print("[polar] Empty path; skipping polar visualization")
        return
    t_for_heat = path[0][0]
    C = compute_cost_map(F, cost_model, t_for_heat, vset)
    C[F.forbid_mask[t_for_heat] > 0.5] = np.nan
    vmin = float(np.nanpercentile(C, 5))
    vmax = float(np.nanpercentile(C, 95))

    fig = plt.figure(figsize=(9, 9))
    stereo_proj = ccrs.NorthPolarStereo(central_longitude=0)
    ax: Any = fig.add_subplot(1, 1, 1, projection=stereo_proj)
    data_crs = ccrs.PlateCarree()

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], data_crs)
    ax.set_title(f"Optimized Arctic Route (Day {t_for_heat})")

    ax.coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='skyblue')

    lats, lons = make_latlon_grids(config.GRID_Y, config.GRID_X, lat_min, lat_max, lon_min, lon_max)
    lon_2d, lat_2d = np.meshgrid(lons, lats)

    im = ax.pcolormesh(
        lon_2d,
        lat_2d,
        C,
        transform=data_crs,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        alpha=0.75,
        shading='auto',
    )

    if path:
        lats_of_path = np.array([lats[y] for _, y, _ in path])
        lons_of_path = np.array([lons[x] for _, _, x in path])
        ax.plot(
            lons_of_path,
            lats_of_path,
            color='cyan',
            linewidth=3,
            marker='o',
            markersize=4,
            transform=data_crs,
            label='Optimized Route',
        )

    fig.colorbar(im, ax=ax, orientation='vertical', label='Edge cost (a.u.)', pad=0.05)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)


def latlon_to_grid_indices(lat: float, lon: float) -> Tuple[int, int]:
    """Convert latitude/longitude to grid indices (y, x)."""
    lat_clamped = max(min(lat, config.LAT_MAX), config.LAT_MIN)
    lon_clamped = max(min(lon, config.LON_MAX), config.LON_MIN)
    lat_span = config.LAT_MAX - config.LAT_MIN
    lon_span = config.LON_MAX - config.LON_MIN
    if lat_span == 0 or lon_span == 0:
        raise ValueError("Invalid lat/lon span in config")
    y = int(round((lat_clamped - config.LAT_MIN) / lat_span * (config.GRID_Y - 1)))
    x = int(round((lon_clamped - config.LON_MIN) / lon_span * (config.GRID_X - 1)))
    y = max(0, min(config.GRID_Y - 1, y))
    x = max(0, min(config.GRID_X - 1, x))
    return y, x


def summarize_path_costs(
    phys: PhysicalCost,
    F: EnvFields,
    path: List[Tuple[int, int, int]],
    speeds: List[float],
) -> pd.DataFrame:
    if not path or not speeds:
        return pd.DataFrame({"Metric": ["(no path)"], "Value": ["-"], "Unit": ["-"]})
    totals = {"CO2e": 0.0, "BC": 0.0, "Noise": 0.0, "Risk": 0.0, "Forbidden": 0}
    for (t0, y0, x0), (t1, y1, x1), v in zip(path[:-1], path[1:], speeds):
        breakdown = phys.edge_cost(2.7, float(v), F, t0, y1, x1)
        totals["CO2e"] += breakdown * phys.w.w_fuel
        totals["BC"] += breakdown * phys.w.w_bc
        totals["Risk"] += breakdown * phys.w.w_risk
        if breakdown > phys.w.bigM * 0.5:
            totals["Forbidden"] += 1
    df = pd.DataFrame([
        ("CO2e (proxy)", totals["CO2e"], "a.u."),
        ("Black Carbon (proxy)", totals["BC"], "a.u."),
        ("Noise (proxy)", totals["Noise"], "a.u."),
        ("Risk (unitless)", totals["Risk"], "a.u."),
        ("Forbidden Hits", totals["Forbidden"], "count"),
    ], columns=["Metric", "Value", "Unit"])
    return df
