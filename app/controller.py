from typing import Annotated
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import Response
import asyncio
import json

from app.schemas import RouteRequest, RouteResponse
from app.services import Service

api_router = APIRouter()


@api_router.post("/api/v1/find_route")
async def find_optimal_route(
    route_request: RouteRequest,
    service: Annotated[Service, Depends()]
) -> Response:
    try:
        result = await asyncio.to_thread(service.get_optimal_route, route_request)
        cost_summary_json = json.dumps(result.cost_summary, ensure_ascii=False)
        print(f"[Controller] Cost summary: {cost_summary_json}")
        return Response(
            content=result.visualization_image,
            media_type="image/png",
            headers={
                "X-Cost-Summary": cost_summary_json
            }
        )
    except HTTPException:
        raise  # HTTPException은 그대로 전파
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))