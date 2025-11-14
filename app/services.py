from typing import List
from fastapi.responses import FileResponse

import torch
import os

from app.schemas import RouteRequest, RouteResponse
from models import (
    EnvFields,
    PhysicalCost,
    VesselSpec,
    CostWeights,
    CostParams,
    TinyNCF,
    NeuralCostFieldDL,
)
from utils import (
    load_processed_env_fields,
    astar_route_with_speeds,
    plot_route_on_polar_stereo,
    summarize_path_costs,
    latlon_to_grid_indices,
)
import config

from app.exceptions import CoordinateNotAllowException, PathNotFoundException

class Service:
    def __init__(self) -> None:
        self.F: EnvFields = load_processed_env_fields()
        vessel = VesselSpec()
        phys = PhysicalCost(vessel, CostWeights(), CostParams())
        self.cost_model = NeuralCostFieldDL(self._load_model(), phys=phys)

    def _load_model(self) -> TinyNCF:
        model = TinyNCF()
        if not config.MODEL_SAVE_PATH.exists():
            raise RuntimeError(f"Model weights missing at {config.MODEL_SAVE_PATH}")
        weights = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(weights)
        model.eval()
        return model

    def get_optimal_route(self, req: RouteRequest) -> RouteResponse:
        start_y, start_x = latlon_to_grid_indices(req.lat_start, req.lon_start)
        goal_y, goal_x = latlon_to_grid_indices(req.lat_goal, req.lon_goal)
        start_node = (req.t_start_idx, start_y, start_x)
        goal_node = (req.t_goal_idx, goal_y, goal_x)
        if not (0 <= start_x < config.GRID_X and 0 <= start_y < config.GRID_Y and 0 <= goal_x <= config.GRID_X and 0 <= goal_y <= config.GRID_Y):
            raise CoordinateNotAllowException()
        print(f"[Service] Finding route: {start_node} -> {goal_node}")
        phys = self.cost_model.phys
        if phys is None:
            raise RuntimeError("Physical cost engine missing")
        old_bcf = phys.vessel.bcf
        old_w_fuel = phys.w.w_fuel
        old_w_bc = phys.w.w_bc
        old_w_risk = phys.w.w_risk
        old_w_fuel_type = phys.vessel.fuel_type
        phys.vessel.fuel_type = req.fuel_type
        phys.vessel.bcf = req.BCF
        phys.w.w_fuel = req.w_fuel
        phys.w.w_bc = req.w_bc
        phys.w.w_risk = req.w_risk
        try:
            path, speeds, total_cost = astar_route_with_speeds(self.F, self.cost_model, start_node, goal_node)
        finally:
            phys.vessel.bcf = old_bcf
            phys.w.w_fuel = old_w_fuel
            phys.w.w_bc = old_w_bc
            phys.w.w_risk = old_w_risk
        if not path or speeds is None:
            raise PathNotFoundException()
        vis_path = str(config.DEFAULT_VIS_PATH)
        try:
            plot_route_on_polar_stereo(self.F, self.cost_model, path, vis_path)
        except Exception as exc:
            print(f"[Service] Visualization failed: {exc}")
            vis_path = ""
        phys = self.cost_model.phys
        if phys is None:
            raise RuntimeError("Physical cost engine missing")
        summary = summarize_path_costs(phys, self.F, path, speeds)
        return RouteResponse(
            visualization_file=vis_path,
            cost_summary=[{str(k): v for k, v in row.items()} for row in summary.to_dict("records")],
        )

    def get_image(self, filename: str) -> FileResponse:
        file_path = os.path.join(config.DEFAULT_VIS_PATH)
        return FileResponse(
            filename=filename,
            media_type="image/png",
            path=file_path
        )