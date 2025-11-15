from fastapi import Depends
from typing import Annotated, Optional
from database.connection import get_db_session
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models import Route
import json


class Repository:
    def __init__(self, session: Annotated[Session, Depends(get_db_session)]) -> None:
        self.session = session

    def add_route(
        self,
        t_start_idx: int,
        t_goal_idx: int,
        lat_start: float,
        lon_start: float,
        lat_goal: float,
        lon_goal: float,
        bcf: float,
        fuel_type: str,
        w_fuel: float,
        w_bc: float,
        w_risk: float,
        visualization_image: bytes,
        cost_summary: list,
    ) -> Route:
        """경로 결과를 DB에 저장"""
        route = Route(
            t_start_idx=t_start_idx,
            t_goal_idx=t_goal_idx,
            lat_start=lat_start,
            lon_start=lon_start,
            lat_goal=lat_goal,
            lon_goal=lon_goal,
            bcf=bcf,
            fuel_type=fuel_type,
            w_fuel=w_fuel,
            w_bc=w_bc,
            w_risk=w_risk,
            visualization_image=visualization_image,
            cost_summary_json=json.dumps(cost_summary),
        )
        self.session.add(route)
        self.session.flush()
        return route

    def get_route(
        self,
        t_start_idx: int,
        t_goal_idx: int,
        lat_start: float,
        lon_start: float,
        lat_goal: float,
        lon_goal: float,
        bcf: float,
        fuel_type: str,
        w_fuel: float,
        w_bc: float,
        w_risk: float,
    ) -> Optional[Route]:
        """동일한 파라미터로 이미 계산된 경로가 있는지 조회"""
        get_route_query = select(Route).filter(
            (Route.t_start_idx == t_start_idx)
            & (Route.t_goal_idx == t_goal_idx)
            & (Route.lat_start == lat_start)
            & (Route.lon_start == lon_start)
            & (Route.lat_goal == lat_goal)
            & (Route.lon_goal == lon_goal)
            & (Route.bcf == bcf)
            & (Route.fuel_type == fuel_type)
            & (Route.w_fuel == w_fuel)
            & (Route.w_bc == w_bc)
            & (Route.w_risk == w_risk)
        )
        return self.session.scalar(get_route_query)