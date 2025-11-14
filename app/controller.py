from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import FileResponse
import asyncio

from app.schemas import RouteRequest, RouteResponse
from app.services import Service

api_router = APIRouter()


def get_service(request: Request) -> Service:
    service = getattr(request.app.state, 'service', None)
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service


@api_router.post("/api/v1/find_route")
async def find_optimal_route(
    route_request: RouteRequest,
    request: Request,
    service: Service = Depends(get_service),
) -> RouteResponse:
    try:
        return await asyncio.to_thread(service.get_optimal_route, route_request)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@api_router.get("/api/v1/images/{filename}")
def get_visualization_image(
    filename: str,
    service: Service = Depends(get_service)
) -> FileResponse:
    return service.get_image(filename)