from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Request model for making predictions.

    Attributes:
        departure_coordinates (list[float]): A list containing the latitude and longitude of the departure point.
        destination_coordinates (list[float]): A list containing the latitude and longitude of the arrival point.
        fuel (float)
        black_carbon (float)
        noise (float)
        risk (float)
    """

    departure_coordinates: list[float]
    destination_coordinates: list[float]
    fuel: float
    black_carbon: float
    noise: float
    risk: float
    