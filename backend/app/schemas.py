from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: Dict[str, float]