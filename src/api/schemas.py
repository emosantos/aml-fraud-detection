"""
Pydantic datamodels for API request and response schemas.

-TransactionFeatures: what the caller sends to /predict
-PredictionResponse: what the API returns from /predict
-MetricsResponse: what the API returns from /metrics
etc.
"""

from pydantic import BaseModel, Field, ConfigDict

# Request body - What the caller sends to /predict

class TransactionFeatures(BaseModel):
    # Raw Amount
    amount_usd: float = Field(..., description="Amount of the transaction")

    # Time Features
    hour_sin: float = Field(...)
    hour_cos: float = Field(...)
    day_of_week: int = Field(...,ge=0,le=6)
    dow_sin: float = Field(...)
    dow_cos: float = Field(...)
    is_weekend: int = Field(..., ge=0, le=1)
    is_business_hours: int = Field(..., ge=0, le=1)
    is_unusual_hour: int = Field(..., ge=0, le=1)
    transaction_hour: int = Field(..., ge=0, le=23, description="Raw transaction hour (0-23)")

    # Transaction type flags
    is_cross_border: int = Field(..., ge=0, le=1)
    currency_mismatch: int = Field(..., ge=0, le=1)

    #Account-level aggregates (Deviance from normal account behavior)

    from_account_avg_amount: float = Field(...)
    from_account_std_amount: float = Field(...)
    from_account_min_amount: float = Field(...)
    from_account_max_amount: float = Field(...)
    from_account_total_transactions: int = Field(..., ge=0)
    from_account_total_volume: float = Field(..., ge=0)
    from_unique_counterparties: int = Field(..., ge=0)
    from_account_cross_border_pct: float = Field(..., ge=0, le=1)
    from_account_unusual_hour_pct: float = Field(..., ge=0, le=1)

    # Amount anomaly features
    amount_z_score: float = Field(...)
    amount_percentile: float = Field(..., ge=0, le=1)
    is_round_amount: int = Field(..., ge=0, le=1)
    is_high_value: int = Field(..., ge=0, le=1)

    # Velocity features
    hours_since_last_txn: float = Field(...)
    is_rapid_succession: int = Field(..., ge=0, le=1)
    txns_same_day: int = Field(..., ge=0)

    # Graph / network features
    from_pagerank: float = Field(...)
    to_pagerank: float = Field(...)
    from_out_degree: int = Field(..., ge=0)
    to_in_degree:    int   = Field(..., ge=0)
    pagerank_ratio: float = Field(..., ge=0)

    # Composite risk signal
    suspicious_signal_count: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "amount_usd": 4500.0,
                "hour_sin": -0.866, "hour_cos": -0.5,
                "day_of_week": 6, "dow_sin": -0.782, "dow_cos": 0.623,
                "is_weekend": 1, "is_business_hours": 0, "is_unusual_hour": 1,
                "is_cross_border": 1, "currency_mismatch": 1,
                "from_account_avg_amount": 750.0,
                "from_account_std_amount": 300.0,
                "from_account_min_amount": 50.0,
                "from_account_max_amount": 2000.0,
                "from_account_total_transactions": 42,
                "from_account_total_volume": 31500.0,
                "from_unique_counterparties": 8,
                "from_account_cross_border_pct": 0.6,
                "from_account_unusual_hour_pct": 0.35,
                "amount_z_score": 3.2, "amount_percentile": 0.97,
                "is_round_amount": 1, "is_high_value": 1,
                "hours_since_last_txn": 0.5,
                "is_rapid_succession": 1, "txns_same_day": 7,
                "from_pagerank": 0.0023, "to_pagerank": 0.0091,
                "from_out_degree": 12, "to_in_degree": 45,
                "pagerank_ratio": 0.253,
                "suspicious_signal_count": 5,
            }
        }

# Reponse Schemas

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    fraud_probability: float  # Raw model score 0.0–1.0
    is_laundering:     bool   # True if score >= threshold
    threshold:         float  # Threshold used for the binary decision
    risk_level:        str    # "LOW" / "MEDIUM" / "HIGH"
    model_version:     str


class MetricsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name:       str
    roc_auc:          float
    pr_auc:           float
    f1:               float
    precision:        float
    recall:           float
    threshold:        float
    confusion_matrix: list   # [[TN, FP], [FN, TP]]
    trained_at:       str
    pr_curve:         dict   # precision/recall/thresholds arrays
    roc_curve:        dict   # fpr/tpr/thresholds arrays


class FeatureImportanceResponse(BaseModel):
    features: list[dict]  # [{"feature": "...", "importance": 0.12}, ...]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status:        str   # "ok" or "degraded"
    model_loaded:  bool
    model_version: str