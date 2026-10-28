
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Body, HTTPException

# Load model artifact
MODEL_PATH = Path(__file__).resolve().parent / "house_rent_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_columns = artifact["feature_columns"]
target_column = artifact.get("target_column", "rent")

app = FastAPI(title="House Rent Prediction API")


@app.get("/")
def home():
    return {
        "message": "House Rent Prediction API is running",
        "docs": "/docs",
        "features": "/features"
    }


@app.get("/features")
def get_features():
    return {
        "target_column": target_column,
        "required_features": feature_columns
    }


@app.post("/predict")
def predict(payload: dict[str, Any] = Body(...)):
    missing = [
        column for column in feature_columns
        if column not in payload
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required input features are missing",
                "missing_features": missing
            }
        )

    # Keep the exact column names and order used in training
    input_data = pd.DataFrame(
        [{column: payload[column] for column in feature_columns}],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_rent": round(float(prediction), 2),
        "target_column": target_column
    }
