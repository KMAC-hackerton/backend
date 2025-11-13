# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any

class RouteRequest(BaseModel):
    t_start_idx: int
    y_start: int
    x_start: int
    t_goal_idx: int
    y_goal: int
    x_goal: int

class RouteResponse(BaseModel):
    status: str
    path_nodes: List[Tuple[int, int, int]]
    speeds_kn: List[float]
    total_cost: float
    path_length: int
    visualization_file: str
    cost_summary: List[Dict[str, Any]]