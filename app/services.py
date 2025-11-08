from app.repositories import PredictionRepository
from app.schemas import PredictionRequest

class PredictionService:
    def __init__(self, prediction_repository: PredictionRepository, model=None):
        self.prediction_repository = prediction_repository
        self.model = model

    def predict(self, request: PredictionRequest):
        self.model.predict(**request.model_dump())