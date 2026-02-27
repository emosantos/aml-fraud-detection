"""
AML Fraud Detection - Multi-Model Training
Trains and compares: Logistic Regression, Random Forest, LightGBM.
Saves per-model artifacts + comparison.json. Best model saved as best_model.pkl.
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "gold_layer" / "engineered_features.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Config
TARGET_COL = "is_laundering"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SELECTION_METRIC = "pr_auc"  # metric used to crown the best model

# Non-feature columns to drop before training
DROP_COLS = [
    "transaction_id",
    "transaction_date",
    "processed_timestamp",
    "from_account",
    "to_account",
    "from_bank",
    "to_bank",
    "original_currency",
    "payment_format",
    "original_amount",
]


# Data 
def load_data(path: Path) -> pd.DataFrame:
    log.info(f"Loading features from {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def split_data(df: pd.DataFrame):
    cols_to_drop = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    log.info(f"Class ratio — Negative: {neg:,} | Positive: {pos:,} | {neg/pos:.1f}x imbalance")
    return X_train, X_test, y_train, y_test


# Model Definitions
def build_models(y_train: pd.Series) -> dict:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)

    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=0.1,
                solver="lbfgs",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            n_estimators=500,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# Training
def train_model(name: str, model, X_train, X_test, y_train, y_test):
    log.info(f"Training {name}...")
    if name == "lightgbm":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
    else:
        model.fit(X_train, y_train)
    return model


# Evaluation
def evaluate_model(name: str, model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

    metrics = {
        "model_name": name,
        "threshold": threshold,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "pr_curve": {
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "trained_at": datetime.utcnow().isoformat(),
    }
    log.info(
        f"  {name:25s} | ROC-AUC: {metrics['roc_auc']:.4f} "
        f"| PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f}"
    )
    return metrics


# Feature Importance
def get_feature_importance(name: str, model, feature_names: list) -> list[dict]:
    try:
        if name == "logistic_regression":
            importance = np.abs(model.named_steps["clf"].coef_[0])
        else:
            importance = model.feature_importances_
        return sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(feature_names, importance)],
            key=lambda x: x["importance"],
            reverse=True,
        )
    except Exception as e:
        log.warning(f"Could not extract feature importance for {name}: {e}")
        return []


# Save
def save_artifacts(models, all_metrics, all_fi, best_name, feature_names):
    # Individual model pickles
    for name, model in models.items():
        with open(MODEL_DIR / f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        log.info(f"Saved {name}_model.pkl")

    # Best model as canonical artifact (API loads this)
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(models[best_name], f)
    log.info(f"Best model ({best_name}) → best_model.pkl")

    # Per-model metrics JSON
    for name, metrics in all_metrics.items():
        with open(MODEL_DIR / f"{name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Best model metrics (used by threshold + live scoring pages)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics[best_name], f, indent=2)

    # Comparison summary
    comparison = {
        "best_model": best_name,
        "selection_metric": SELECTION_METRIC,
        "trained_at": datetime.utcnow().isoformat(),
        "models": {
            name: {
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "confusion_matrix": m["confusion_matrix"],
                "trained_at": m["trained_at"],
            }
            for name, m in all_metrics.items()
        },
    }
    with open(MODEL_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("Saved comparison.json")

    # Feature importance for best model
    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump(all_fi[best_name], f, indent=2)

    # Feature names
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info(f"All artifacts saved to {MODEL_DIR}/")


# Main
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Gold features not found at {DATA_PATH}. "
            "Run the feature engineering pipeline first to generate the Gold Layer."
        )
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)
    feature_names = X_train.columns.tolist()

    model_defs = build_models(y_train)
    trained_models, all_metrics, all_fi = {}, {}, {}

    log.info("=" * 60)
    for name, model in model_defs.items():
        trained = train_model(name, model, X_train, X_test, y_train, y_test)
        metrics = evaluate_model(name, trained, X_test, y_test)
        fi = get_feature_importance(name, trained, feature_names)
        trained_models[name] = trained
        all_metrics[name] = metrics
        all_fi[name] = fi

    best_name = max(all_metrics, key=lambda n: all_metrics[n][SELECTION_METRIC])

    log.info("=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)
    for name, m in all_metrics.items():
        tag = " ← BEST" if name == best_name else ""
        log.info(
            f"  {name:25s} | ROC-AUC: {m['roc_auc']:.4f} "
            f"| PR-AUC: {m['pr_auc']:.4f} | F1: {m['f1']:.4f}{tag}"
        )

    save_artifacts(trained_models, all_metrics, all_fi, best_name, feature_names)
    log.info(f"\nDone. Best model: {best_name}")

if __name__ == "__main__":
    main()