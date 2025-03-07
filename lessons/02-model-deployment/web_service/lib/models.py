from pydantic import BaseModel, Field
from typing import Literal


class DiamondData(BaseModel):
    carat: float = Field(..., gt=0, le=10, example=0.5)
    cut: Literal["Fair", "Good", "Very Good", "Premium", "Ideal"] = Field(..., example="Ideal")
    color: Literal["D", "E", "F", "G", "H", "I", "J"] = Field(..., example="E")
    clarity: Literal["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"] = Field(..., example="VS1")
    depth: float = Field(..., gt=0, le=100, example=61.5)
    table: float = Field(..., gt=0, le=100, example=55.0)
    x: float = Field(..., gt=0, le=100, example=5.15)
    y: float = Field(..., gt=0, le=100, example=5.2)
    z: float = Field(..., gt=0, le=100, example=3.2)


class PredictionOut(BaseModel):
    price_prediction: float
