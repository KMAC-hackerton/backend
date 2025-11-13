import numpy as np
import pandas as pd
from .schemas import RouteRequest, RouteResponse
from config import EnvFields, VSET, DEFAULT_SAVE_PATH
from models import NeuralCostFieldDL, PhysicalCost
from app.exceptions import InvalidRequestException, PathNotFoundException
from utils import (
    astar_route_with_speeds, 
    plot_route_visualization, 
    summarize_path_costs
)

class Service:
    def __init__(self):
        pass
    async def get_optimal_route(self, req: RouteRequest, F: EnvFields, 
                      cost_model: NeuralCostFieldDL, 
                      phys: PhysicalCost) -> RouteResponse:
        T, Y, X = F.SIC.shape

        if not (0 <= req.t_start_idx < T and 0 <= req.t_goal_idx < T):
            raise InvalidRequestException()
        if not (0 <= req.x_start < X and 0 <= req.y_start < Y and 
                0 <= req.x_goal < X and 0 <= req.y_goal < Y):
            raise InvalidRequestException()
        if req.t_goal_idx <= req.t_start_idx:
            raise InvalidRequestException()

        start_node = (req.t_start_idx, req.y_start, req.x_start)
        goal_node = (req.t_goal_idx, req.y_goal, req.x_goal)
        print(f"Finding route from {start_node} to {goal_node}...")

        # 2. 핵심 로직: A* 알고리즘 호출
        path, speeds, total_cost = await astar_route_with_speeds(
            F, cost_model, start_node, goal_node, VSET
        )

        if not path:
            print("Route finding failed.")
            raise PathNotFoundException()
        
        print(f"Route found with {len(path)} nodes.")

        save_vis_path = DEFAULT_SAVE_PATH
        try:
            await plot_route_visualization(F, cost_model, path, savepath=save_vis_path)
        except Exception as e:
            print(f"Warning: Visualization failed. {e}")
            save_vis_path = "N/A (Plotting failed)"
        
        # 4. 부가 로직: 비용 요약
        cost_summary_df = await summarize_path_costs(phys, F, path, speeds)
        
        # 안전 처리: speeds 또는 cost_summary_df가 None일 수 있으므로 기본값 부여
        speeds_list = [] if speeds is None else [float(v) for v in speeds]

        if cost_summary_df is None:
            cost_summary_records = []
        else:
            cost_summary_records = [
            {str(k): (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in cost_summary_df.to_dict(orient="records")
            ]

        return RouteResponse(
            status="success",
            path_nodes=path,
            speeds_kn=speeds_list,
            total_cost=float(total_cost),
            path_length=len(path),
            visualization_file=save_vis_path,
            cost_summary=cost_summary_records,
        )