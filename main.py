from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = "churn_pipeline.joblib"

app = FastAPI(
    title="Telecom Churn Predictor API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

# Allow the frontend (any origin) to call this API. Tighten allow_origins
# to your actual frontend domain once deployed, for better security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    age: float = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0, le=200)
    contract_type: Literal["Month-To-Month", "One Year", "Two Year"]
    internet_service: Literal["Fiber", "Dsl", "No"]
    num_addon_services: int = Field(..., ge=0, le=20)
    monthly_charges: float = Field(..., ge=0)
    data_usage_gb: float = Field(..., ge=0)
    support_calls: int = Field(..., ge=0, le=50)
    payment_method: Literal[
        "Mailed Check", "Bank Transfer", "Credit Card", "Electronic Check"
    ]
    total_charges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 45,
                "tenure_months": 24,
                "contract_type": "Month-To-Month",
                "internet_service": "Fiber",
                "num_addon_services": 3,
                "monthly_charges": 75.5,
                "data_usage_gb": 120.0,
                "support_calls": 2,
                "payment_method": "Credit Card",
                "total_charges": 1800.0,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    churn_probability: float


def predict_one(features: dict, bundle: dict) -> dict:
    row = pd.DataFrame([features])
    row["avg_monthly_spend"] = row["total_charges"] / (row["tenure_months"] + 1)
    row["is_new_customer"] = (row["tenure_months"] <= 12).astype(int)

    pipeline = bundle["pipeline"]
    label = int(pipeline.predict(row)[0])
    proba = pipeline.predict_proba(row)[0]
    churn_probability = float(proba[1])
    confidence = float(proba.max())

    return {
        "prediction": label,
        "label": "Likely to churn" if label == 1 else "Likely to stay",
        "confidence": round(confidence, 3),
        "churn_probability": round(churn_probability, 3),
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "telecom-churn-predictor"}


@app.get("/health")
def health():
    bundle = get_bundle()
    return {"status": "healthy", "model_version": bundle.get("model_version", "unknown")}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        bundle = get_bundle()
        result = predict_one(features.model_dump(), bundle)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
