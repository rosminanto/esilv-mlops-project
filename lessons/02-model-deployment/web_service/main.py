from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    MODEL_VERSION,
    PATH_TO_MODEL,
    PATH_TO_PREPROCESSOR_DV,
    PATH_TO_PREPROCESSOR_SCALER,
)
from fastapi import FastAPI

from lib.modelling import run_inference
from lib.models import DiamondData, PredictionOut
from lib.utils import load_model, load_preprocessor_dv, load_preprocessor_scaler

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: DiamondData):
    dv = load_preprocessor_dv(PATH_TO_PREPROCESSOR_DV)
    scaler = load_preprocessor_scaler(PATH_TO_PREPROCESSOR_SCALER)
    model = load_model(PATH_TO_MODEL)
    price = run_inference([payload], dv, scaler, model)
    return {"price_prediction": price[0]}
