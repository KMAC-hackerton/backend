from fastapi import APIRouter, Request, Depends
import numpy as np
import asyncio

from typing import Annotated
from app.schemas import RouteRequest, RouteResponse
from app.services import Service

api_router = APIRouter()

@api_router.post("/api/v1/find_route")
async def find_optimal_route(
    route_request: RouteRequest,
    request: Request,
    service: Annotated[Service, Depends()],
):
    """비동기 엔드포인트: 내부의 동기적 경로 탐색은 스레드 풀로 옮겨 실행합니다."""
    F = request.app.state.env_fields
    cost_model = request.app.state.ncf_cost_model
    phys_cost = request.app.state.phys_cost

    # Service.get_optimal_route는 CPU 바운드(동기) 함수이므로
    # asyncio.to_thread로 스레드풀에서 실행해 비동기 엔드포인트에서 await 가능하도록 처리.
    result = await asyncio.to_thread(
        service.get_optimal_route,
        req=route_request,
        F=F,
        cost_model=cost_model,
        phys=phys_cost,
    )

    return result