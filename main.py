"""FastAPI endpoints for the trained video game sales models."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
regression_model = joblib.load(MODELS / "regression.joblib")
classification_model = joblib.load(MODELS / "classification.joblib")

app = FastAPI(title="Video Game Sales ML Demo")


class GameInput(BaseModel):
    platform: str = Field(min_length=1)
    genre: str = Field(min_length=1)
    year: float = Field(ge=1980, le=2035)


def model_input(data: GameInput):
    return pd.DataFrame([{
        "Platform": data.platform,
        "Genre": data.genre,
        "Year": data.year,
    }])


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.post("/predict/regression")
def predict_regression(data: GameInput):
    prediction = regression_model.predict(model_input(data))[0]
    return {"prediction": round(float(prediction), 3), "unit": "million copies"}


@app.post("/predict/classification")
def predict_classification(data: GameInput):
    prediction = classification_model.predict(model_input(data))[0]
    return {"prediction": str(prediction)}
