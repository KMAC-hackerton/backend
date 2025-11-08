from fastapi import APIRouter
from fastapi import Depends
from typing import Annotated
from app.schemas import PredictionRequest, PredictionResponse
from app.services import PredictionService

api_router = APIRouter()

@api_router.get("/health")
async def health_check():
    return {"status": "ok"}

@api_router.get("/predict")
async def predict(
    request: PredictionRequest
    service: Annotated[PredictionService, Depends()]
) -> PredictionResponse:
    prediction = 
    return PredictionResponse(prediction="This is a dummy prediction")