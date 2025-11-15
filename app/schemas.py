from pydantic import BaseModel
from typing import List, Dict, Any


class RouteRequest(BaseModel):
    t_start_idx: int
    t_goal_idx: int
    lat_start: float
    lon_start: float
    lat_goal: float
    lon_goal: float
    BCF: float
    fuel_type: str
    w_fuel: float
    w_bc: float
    w_risk: float


class RouteResponse(BaseModel):
    visualization_image: bytes  # PNG 이미지 바이트
    cost_summary: List[Dict[str, Any]]