"""
FastAPI application for the API endpoints.

start the server with:
uvicorn src.api.main:app --reload

ENDPOINTS:
GET /health - Health check endpoint
POST /predict - score a transaction for fraud risk
GET /metrics - return model performance metrics
GET /metrics{model}
GET /comparison - return performance comparison between models
GET /features - return feature importance scores
"""

import json
import pickle
import logging
from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    TransactionFeatures, 
    PredictionResponse, MetricsResponse, 
    FeatureImportanceResponse,
    HealthResponse
)

log = logging.getLogger(__name__)

# Paths

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

MODEL_VERSION = "multi-model-v1"  # Update as needed

# App

app = FastAPI(
    title="AML Fraud Detection API",
    description="Real-time Anti-Money Laundering fraud scoring",
    version="1.0.0",
    docs_url="/docs", # Swagger UI
    redoc_url="/redoc" # ReDoc UI
)

# Corrs middleware (Allows browser-based clients to call the API from a different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_MODELS = ["logistic_regression", "random_forest", "lightgbm"]

# Model Loading

#lru_cache to keep the model in memory after first load

@lru_cache(maxsize=1)
def load_model():
    """Load the best model (chosen by PR-AUC during training)."""
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"best_model.pkl not found at {path}. "
            "Run `python -m src.models.train` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    """Load the best model's full metrics (includes curve data)."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("metrics.json not found. Run training first.")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_comparison() -> dict:
    """Load the side-by-side comparison of all 3 models."""
    path = MODEL_DIR / "comparison.json"
    if not path.exists():
        raise FileNotFoundError("comparison.json not found. Run training first.")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_feature_importance() -> list:
    path = MODEL_DIR / "feature_importance.json"
    if not path.exists():
        raise FileNotFoundError("feature_importance.json not found.")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_feature_names() -> list:
    path = MODEL_DIR / "feature_names.json"
    if not path.exists():
        raise FileNotFoundError("feature_names.json not found.")
    with open(path) as f:
        return json.load(f)


def load_metrics_by_name(name: str) -> dict:
    # Not cached because there are multiple models
    path = MODEL_DIR / f"{name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{name}_metrics.json not found.")
    with open(path) as f:
        return json.load(f)


def risk_level(prob: float) -> str:
    """Convert a raw probability to a human-readable risk band."""
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    return "HIGH"

# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check whether the model is loaded and the API is ready."""
    try:
        load_model()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(
    transaction: TransactionFeatures,
    # Query parameter — caller can pass ?threshold=0.3 in the URL
    # Default is 0.5, must be between 0 and 1
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    """
    Score a single transaction.

    Returns a fraud probability and a binary decision based on the threshold.
    Lower the threshold to catch more fraud (higher recall, lower precision).
    Raise it to reduce false alarms (higher precision, lower recall).
    """
    try:
        model        = load_model()
        feature_names = load_feature_names()
    except FileNotFoundError as e:
        # 503 = Service Unavailable — the model isn't ready yet
        raise HTTPException(status_code=503, detail=str(e))

    # Convert the Pydantic object to a dict, then to a DataFrame.
    # We reorder columns to match exactly what the model was trained on.
    features = transaction.model_dump()
    X = pd.DataFrame([features])[feature_names]

    # predict_proba returns [[prob_class_0, prob_class_1]]
    # [:, 1] takes the probability of the positive class (laundering)
    prob = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        is_laundering=prob >= threshold,
        threshold=threshold,
        risk_level=risk_level(prob),
        model_version=MODEL_VERSION,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Model"])
def get_metrics():
    """Evaluation metrics for the best model (full curve data included)."""
    try:
        return MetricsResponse(**load_metrics())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics/{model_name}", response_model=MetricsResponse, tags=["Model"])
def get_model_metrics(model_name: str):
    """Evaluation metrics for a specific model by name."""
    if model_name not in VALID_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Valid options: {VALID_MODELS}"
        )
    try:
        return MetricsResponse(**load_metrics_by_name(model_name))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/comparison", tags=["Model"])
def get_comparison():
    """Side-by-side summary of all trained models."""
    try:
        return load_comparison()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/features", response_model=FeatureImportanceResponse, tags=["Model"])
def get_feature_importance(
    top_n: int = Query(default=20, ge=1, le=100)
):
    """Feature importances from the best model, sorted descending."""
    try:
        fi = load_feature_importance()
        return FeatureImportanceResponse(features=fi[:top_n])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/", tags=["System"])
def root():
    return {
        "name": "AML Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

