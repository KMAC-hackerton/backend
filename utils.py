import heapq
import math
from typing import Tuple, List, Optional, Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from models import EnvFields, PhysicalCost, NeuralCostFieldDL

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
    max_iterations: int = 50000,
) -> Tuple[Optional[List[Tuple[int, int, int]]], Optional[List[float]], float]:
    T, Y, X = F.SIC.shape
    (t0, y0, x0), (tG, yG, xG) = start, goal

    def in_bounds(t, y, x):
        return 0 <= t < T and 0 <= y < Y and 0 <= x < X

    def heuristic(t, y, x):
        spatial = abs(y - yG) + abs(x - xG)
        time_dist = max(0, tG - t)
        return spatial * d_cell_nm * 0.5 + time_dist * 2.0

    pq = []
    heapq.heappush(pq, (heuristic(t0, y0, x0), 0.0, (t0, y0, x0)))
    g_cost = {(t0, y0, x0): 0.0}
    parent = {}
    best_speed = {}
    visited = set()
    iteration = 0

    while pq:
        iteration += 1
        if iteration > max_iterations:
            print(f"⚠️ A* exceeded {max_iterations} iterations")
            return None, None, float('inf')
        if iteration % 5000 == 0:
            print(f"[A*] Iter {iteration}, queue={len(pq)}, visited={len(visited)}")
        _, gc, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        t, y, x = node
        if node == (tG, yG, xG):
            path = [node]
            speeds = []
            while node in parent:
                speeds.append(best_speed[node])
                node = parent[node]
                path.append(node)
            return list(reversed(path)), list(reversed(speeds)), gc
        nt = t + 1
        if nt >= T:
            continue
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            ny, nx = y + dy, x + dx
            if not in_bounds(nt, ny, nx):
                continue
            nnode = (nt, ny, nx)
            if nnode in visited:
                continue
            best_c = float('inf')
            best_v = None
            for v in vset:
                c = cost_model.predict(d_cell_nm, float(v), F, nt, ny, nx)
                if c < best_c:
                    best_c = c
                    best_v = v
            ng = gc + best_c
            if ng < g_cost.get(nnode, float('inf')):
                g_cost[nnode] = ng
                parent[nnode] = node
                best_speed[nnode] = best_v
                heapq.heappush(pq, (ng + heuristic(nt, ny, nx), ng, nnode))
    return None, None, float('inf')


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
