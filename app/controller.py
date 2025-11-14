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

    print(f"[CONTROLLER] Received request: start=({route_request.t_start_idx},{route_request.y_start},{route_request.x_start}), "
          f"goal=({route_request.t_goal_idx},{route_request.y_goal},{route_request.x_goal})")

    # Service.get_optimal_route는 CPU 바운드(동기) 함수이므로
    # asyncio.to_thread로 스레드풀에서 실행해 비동기 엔드포인트에서 await 가능하도록 처리.
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                service.get_optimal_route,
                req=route_request,
                F=F,
                cost_model=cost_model,
                phys=phys_cost,
            ),
            timeout=300.0  # 5분 타임아웃
        )
        print("[CONTROLLER] ✅ Request completed successfully")
        return result
    except asyncio.TimeoutError:
        print("[CONTROLLER] ❌ Request timed out after 300s")
        return RouteResponse(
            status="error",
            path_nodes=[],
            speeds_kn=[],
            total_cost=0.0,
            path_length=0,
            visualization_file="N/A",
            cost_summary=[{"error": "Request timed out - route computation took too long"}]
        )
    except Exception as e:
        print(f"[CONTROLLER] ❌ Error during route computation: {e}")
        raise