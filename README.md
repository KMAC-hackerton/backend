# backend

## Polar Visualization Helper

- `plot_route_on_polar_stereo` (in `utils.py`) renders the grid and the chosen route on a North Polar Stereographic map using `cartopy`. Pass the loaded environment, trained cost model, and the path returned by the service, along with a file path to save the PNG.
- Cartopy is optional; the function will skip rendering if `cartopy` is not installed. Install it manually with `pip install cartopy` before invoking the helper to ensure the polar projection works.

## Request Parameters

- `BCF` (ice class factor): incoming `RouteRequest` payloads must provide this float so the service can adjust the `VesselSpec` ice-breaking coefficient before computing the route. This influences both physical and DL cost components in `models.py` and lets callers experiment with different ice-handling assumptions.
- `fuel_type`: string identifier (e.g., `MGO`) that defines the emissions factors used inside `PhysicalCost`.
- `w_fuel`, `w_bc`, `w_risk`: cost-weight scalars applied to the corresponding emissions/risk proxies; these allow front-end callers to trade off between fuel, BC, and risk penalties when planning the route.