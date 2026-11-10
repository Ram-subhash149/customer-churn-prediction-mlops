from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best_xgboost_model.pkl"))
METADATA_PATH = ROOT / "artifacts" / "model_metrics.json"
DEFAULT_THRESHOLD = 0.50
OPTIMAL_THRESHOLD = 0.5503

FEATURES = [
    "Gender", "Age", "Scholarship", "Hypertension", "Diabetes",
    "Alcoholism", "Handicap", "SMS_received", "WaitingDays",
    "ScheduledWeekday", "AppointmentWeekday", "ScheduledHour",
    "PriorAppointments", "PriorNoShows", "PriorNoShowRate"
]

app = FastAPI(
    title="Medical Appointment No-Show Prediction API",
    description=(
        "Predicts the probability that a patient will miss a scheduled appointment. "
        "Prediction is a decision-support estimate, not a clinical decision."
    ),
    version="1.0.0",
)

model: XGBClassifier | None = None

@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

class AppointmentInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "Gender": "F",
            "Age": 35,
            "Scholarship": 0,
            "Hypertension": 0,
            "Diabetes": 0,
            "Alcoholism": 0,
            "Handicap": 0,
            "SMS_received": 1,
            "ScheduledDay": "2016-05-01T10:30:00",
            "AppointmentDay": "2016-05-10T00:00:00",
            "PriorAppointments": 2,
            "PriorNoShows": 1,
            "threshold": 0.5503
        }
    })

    Gender: str = Field(..., pattern="^(F|M|Female|Male)$")
    Age: int = Field(..., ge=0, le=120)
    Scholarship: int = Field(..., ge=0, le=1)
    Hypertension: int = Field(..., ge=0, le=1)
    Diabetes: int = Field(..., ge=0, le=1)
    Alcoholism: int = Field(..., ge=0, le=1)
    Handicap: int = Field(..., ge=0, le=4)
    SMS_received: int = Field(..., ge=0, le=1)
    ScheduledDay: str
    AppointmentDay: str
    PriorAppointments: int = Field(0, ge=0)
    PriorNoShows: int = Field(0, ge=0)
    threshold: float = Field(OPTIMAL_THRESHOLD, ge=0, le=1)

class BatchPredictionRequest(BaseModel):
    appointments: List[AppointmentInput] = Field(..., min_length=1, max_length=500)

def _build_features(item: AppointmentInput) -> pd.DataFrame:
    try:
        scheduled = pd.to_datetime(item.ScheduledDay, utc=True, errors="raise")
        appointment = pd.to_datetime(item.AppointmentDay, utc=True, errors="raise")
    except Exception as exc:
        raise ValueError("ScheduledDay and AppointmentDay must be valid date/time strings.") from exc

    waiting_days = int((appointment.normalize() - scheduled.normalize()).days)
    if waiting_days < 0:
        raise ValueError("AppointmentDay must be on or after ScheduledDay.")
    if item.PriorNoShows > item.PriorAppointments:
        raise ValueError("PriorNoShows cannot exceed PriorAppointments.")

    gender = 0 if item.Gender.lower() in ("f", "female") else 1
    prior_rate = (
        item.PriorNoShows / item.PriorAppointments
        if item.PriorAppointments > 0 else 0.0
    )
    row = {
        "Gender": gender,
        "Age": item.Age,
        "Scholarship": item.Scholarship,
        "Hypertension": item.Hypertension,
        "Diabetes": item.Diabetes,
        "Alcoholism": item.Alcoholism,
        "Handicap": item.Handicap,
        "SMS_received": item.SMS_received,
        "WaitingDays": waiting_days,
        "ScheduledWeekday": int(scheduled.dayofweek),
        "AppointmentWeekday": int(appointment.dayofweek),
        "ScheduledHour": int(scheduled.hour),
        "PriorAppointments": item.PriorAppointments,
        "PriorNoShows": item.PriorNoShows,
        "PriorNoShowRate": float(prior_rate),
    }
    return pd.DataFrame([row], columns=FEATURES)

def _predict_one(item: AppointmentInput) -> dict:
    if model is None:
        raise RuntimeError("Prediction model has not been loaded.")
    features = _build_features(item)
    probability = float(model.predict_proba(features)[0, 1])
    threshold = float(item.threshold)
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "prediction_label": "No-show risk" if prediction == 1 else "Likely to attend",
        "no_show_probability": round(probability, 6),
        "show_probability": round(1.0 - probability, 6),
        "threshold": threshold,
        "risk_level": (
            "High" if probability >= 0.70 else
            "Moderate" if probability >= threshold else
            "Low"
        )
    }

@app.get("/", tags=["General"])
def root():
    return {
        "project": "Medical Appointment No-Show Prediction",
        "registration_number": "241FA18343",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", tags=["General"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metadata", tags=["General"])
def metadata():
    result = {
        "model": "Tuned XGBoost",
        "target": "Target (1 = No-show, 0 = Show)",
        "features": FEATURES,
        "default_threshold": OPTIMAL_THRESHOLD,
        "model_file": str(MODEL_PATH.name)
    }
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
            result["reported_metrics"] = metrics
        except Exception:
            pass
    return result

@app.post("/predict", tags=["Prediction"])
def predict(payload: AppointmentInput):
    try:
        return _predict_one(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(payload: BatchPredictionRequest):
    try:
        return {
            "count": len(payload.appointments),
            "predictions": [_predict_one(item) for item in payload.appointments]
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")
