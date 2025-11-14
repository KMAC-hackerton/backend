from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional


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
    visualization_file: str  # 이미지 파일 경로 또는 Base64 인코딩된 문자열
    cost_summary: List[Dict[str, Any]]


class RouteResponseWithBase64(BaseModel):
    """Base64로 인코딩된 이미지를 포함하는 응답"""
    visualization_base64: str  # data:image/png;base64,... 형식
    visualization_filename: str  # 원본 파일명
    cost_summary: List[Dict[str, Any]]


class RouteResponseWithURL(BaseModel):
    """이미지 URL을 포함하는 응답"""
    visualization_url: str  # 이미지 접근 URL
    visualization_filename: str  # 파일명
    cost_summary: List[Dict[str, Any]]