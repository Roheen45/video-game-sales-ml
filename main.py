"""FastAPI endpoints for the video game sales models."""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

REGRESSION_MODELS = {
    name: joblib.load(MODEL_DIR / f"regression_{name}.joblib")
    for name in ("linear", "multiple", "polynomial", "lasso")
}
CLASSIFICATION_MODELS = {
    name: joblib.load(MODEL_DIR / f"classification_{name}.joblib")
    for name in ("logistic", "random_forest", "decision_tree", "naive_bayes", "svm", "tfidf")
}
METRICS = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))

app = FastAPI(title="Video Game Sales ML Demo")


class RegressionInput(BaseModel):
    model: Literal["linear", "multiple", "polynomial", "lasso"] = "multiple"
    na_sales: float = Field(ge=0)
    eu_sales: float = Field(ge=0)
    jp_sales: float = Field(ge=0)
    other_sales: float = Field(ge=0)


class ClassificationInput(BaseModel):
    model: Literal["logistic", "random_forest", "decision_tree", "naive_bayes", "svm", "tfidf"] = "random_forest"
    platform: str = "PS4"
    year: float = Field(default=2016, ge=1980, le=2020)
    name: str = ""


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/metrics")
def get_metrics():
    return METRICS


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    features = ["NA_Sales"] if data.model == "linear" else [
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"
    ]
    values = {
        "NA_Sales": data.na_sales,
        "EU_Sales": data.eu_sales,
        "JP_Sales": data.jp_sales,
        "Other_Sales": data.other_sales,
    }
    row = pd.DataFrame([[values[feature] for feature in features]], columns=features)
    prediction = REGRESSION_MODELS[data.model].predict(row)[0]
    return {"prediction": round(max(0.0, float(prediction)), 3), "unit": "million copies"}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    if data.model == "tfidf":
        if not data.name.strip():
            raise HTTPException(status_code=422, detail="Enter a game title for the TF-IDF model.")
        prediction = CLASSIFICATION_MODELS["tfidf"].predict([data.name.strip()])[0]
    else:
        row = pd.DataFrame([{"Platform": data.platform, "Year": data.year}])
        prediction = CLASSIFICATION_MODELS[data.model].predict(row)[0]
    return {"prediction": str(prediction)}
