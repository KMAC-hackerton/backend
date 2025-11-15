from typing import List
from fastapi.responses import FileResponse
from fastapi import Depends
from typing import Annotated
import io
import json

import torch
import os

from app.schemas import RouteRequest, RouteResponse
from app.repositoires import Repository
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
    def __init__(self, repository: Annotated[Repository, Depends()]) -> None:
        self.F: EnvFields = load_processed_env_fields()
        vessel = VesselSpec()
        phys = PhysicalCost(vessel, CostWeights(), CostParams())
        self.cost_model = NeuralCostFieldDL(self._load_model(), phys=phys)
        self.repository = repository

    def _load_model(self) -> TinyNCF:
        model = TinyNCF()
        if not config.MODEL_SAVE_PATH.exists():
            raise RuntimeError(f"Model weights missing at {config.MODEL_SAVE_PATH}")
        weights = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(weights)
        model.eval()
        return model

    def get_optimal_route(self, req: RouteRequest) -> RouteResponse:
        # 1. DB에서 캐시된 결과 조회
        cached_route = self.repository.get_route(
            t_start_idx=req.t_start_idx,
            t_goal_idx=req.t_goal_idx,
            lat_start=req.lat_start,
            lon_start=req.lon_start,
            lat_goal=req.lat_goal,
            lon_goal=req.lon_goal,
            bcf=req.BCF,
            fuel_type=req.fuel_type,
            w_fuel=req.w_fuel,
            w_bc=req.w_bc,
            w_risk=req.w_risk,
        )
        
        if cached_route is not None:
            print(f"[Service] Cache hit! Returning cached route (id={cached_route.id})")
            return RouteResponse(
                visualization_image=cached_route.visualization_image,
                cost_summary=json.loads(cached_route.cost_summary_json),
            )

        # 2. 캐시 미스 - 새로 계산
        print(f"[Service] Cache miss. Computing new route...")
        start_y, start_x = latlon_to_grid_indices(req.lat_start, req.lon_start)
        goal_y, goal_x = latlon_to_grid_indices(req.lat_goal, req.lon_goal)
        
        start_node = (req.t_start_idx, start_y, start_x)
        goal_node = (req.t_goal_idx, goal_y, goal_x)
        
        if not (
            0 <= start_x < config.GRID_X
            and 0 <= start_y < config.GRID_Y
            and 0 <= goal_x < config.GRID_X
            and 0 <= goal_y < config.GRID_Y
        ):
            raise CoordinateNotAllowException()
        
        print(f"[Service] Finding route: {start_node} -> {goal_node}")
        
        # 물리 파라미터 임시 변경
        phys = self.cost_model.phys
        if phys is None:
            raise RuntimeError("Physical cost engine missing")
        
        old_bcf = phys.vessel.bcf
        old_w_fuel = phys.w.w_fuel
        old_w_bc = phys.w.w_bc
        old_w_risk = phys.w.w_risk
        old_fuel_type = phys.vessel.fuel_type
        
        phys.vessel.fuel_type = req.fuel_type
        phys.vessel.bcf = req.BCF
        phys.w.w_fuel = req.w_fuel
        phys.w.w_bc = req.w_bc
        phys.w.w_risk = req.w_risk
        
        try:
            path, speeds, total_cost = astar_route_with_speeds(
                self.F, self.cost_model, start_node, goal_node
            )
        finally:
            # 원래 파라미터로 복원
            phys.vessel.bcf = old_bcf
            phys.w.w_fuel = old_w_fuel
            phys.w.w_bc = old_w_bc
            phys.w.w_risk = old_w_risk
            phys.vessel.fuel_type = old_fuel_type
        
        if not path or speeds is None:
            raise PathNotFoundException()
        
        # 시각화 생성
        vis_path = str(config.OUTDIR / f"route_{req.t_start_idx}_{req.t_goal_idx}.png")
        try:
            plot_route_on_polar_stereo(self.F, self.cost_model, path, vis_path)
        except Exception as exc:
            print(f"[Service] Visualization failed: {exc}")
            vis_path = ""
        
        # 비용 요약
        summary = summarize_path_costs(phys, self.F, path, speeds)
        summary_rows = [
            row
            for row in summary.to_dict("records")
            if row.get("Metric") not in {"Noise (proxy)", "Forbidden Hits"}
        ]
        cost_summary_clean = [{str(k): v for k, v in row.items()} for row in summary_rows]
        
        # 이미지를 바이트로 읽기
        with open(vis_path, "rb") as f:
            image_bytes = f.read()
        
        # DB에 저장
        self.repository.add_route(
            t_start_idx=req.t_start_idx,
            t_goal_idx=req.t_goal_idx,
            lat_start=req.lat_start,
            lon_start=req.lon_start,
            lat_goal=req.lat_goal,
            lon_goal=req.lon_goal,
            bcf=req.BCF,
            fuel_type=req.fuel_type,
            w_fuel=req.w_fuel,
            w_bc=req.w_bc,
            w_risk=req.w_risk,
            visualization_image=image_bytes,
            cost_summary=cost_summary_clean,
        )
        
        print(f"[Service] Route saved to DB")
        
        return RouteResponse(
            visualization_image=image_bytes,
            cost_summary=cost_summary_clean,
        )