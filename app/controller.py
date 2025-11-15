from typing import Annotated
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import Response
import asyncio

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
        return Response(
            content=result.visualization_image,
            media_type="image/png",
            headers={
                "X-Cost-Summary": str(result.cost_summary)
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))