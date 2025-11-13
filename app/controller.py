from fastapi import APIRouter, Request
import numpy as np

from app.schemas import RouteRequest, RouteResponse
from app.services import Service

api_router = APIRouter()

@api_router.post("/api/v1/find_route")
async def find_optimal_route(route_request: RouteRequest, request: Request, service: Service):
    F = request.app.state.env_fields
    cost_model = request.app.state.ncf_cost_model
    phys_cost = request.app.state.phys_cost
    result = await service.get_optimal_route(
        req=route_request, 
        F=F, 
        cost_model=cost_model, 
        phys=phys_cost
    )
    
    return result