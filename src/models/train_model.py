import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    auc,
    roc_curve,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0,str(project_root))

# Configuration

# Features from feature_engineering.py

FEATURE_COLUMNS = [
    # Temporal
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_business_hours', 'is_unusual_hour',

    # Amount
    'amount_usd', 'amount_z_score', 'amount_percentile',
    'is_round_amount', 'is_high_value',
    'account_avg_amount', 'account_std_amount',
    'account_total_transactions', 'account_total_volume',

    # Velocity
    'hours_since_last_txn', 'is_rapid_succession',
    'txns_same_day', 'unique_counterparties',

    # Network
    'from_pagerank', 'to_pagerank',
    'from_out_degree', 'to_in_degree', 'pagerank_ratio',

    # Behavioral
    'is_cross_border', 'currency_mismatch',
    'account_cross_border_pct', 'account_unusual_hour_pct',
    'suspicious_signal_count'
]

TARGET_COLUMN = 'is_laundering'

# MLflow setup

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = 'aml_fraud_detection'

# Data preparation

class DataPreparer:
    """
    Handles feature seelction, splitting, imbalance handling and scaling
    """
    def __init__(self, test_size=0.2,random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load_features(self):
        """
        Load engineered features
        """
        path = project_root / 'data' / 'features' / 'engineered_features.parquet'
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}.\n"
                                f"Run feature_engineering.py first.")
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} transactions with {len(df.columns)} features")
        return df

    def select_and_validate(self, df):
        """
        Select only the features we want and validate they exist.
        """
        missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f" Missing columns: {missing}")

        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMN].astype(int).copy()

        # Fill any NaN values
        X = X.fillna(0)

        print(f"   Features selected: {len(FEATURE_COLUMNS)} columns")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        print(f"   Fraud rate: {y.mean()*100:.2f}%")

        return X, y

    def split_data(self, X, y):
        """
        Split into train/test with STRATIFICATION. (both train and test have the same fraud ratio)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Keep same fraud ratio in train and test
        )

        print(f"   Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
        print(f"   Train fraud rate: {y_train.mean()*100:.2f}%")
        print(f"   Test fraud rate:  {y_test.mean()*100:.2f}%")

        return X_train, X_test, y_train, y_test

    def handle_imbalance(self, X_train, y_train):
        """
        SMOTE (Synthetic Minority Over-sampling Technique)

        HOW SMOTE WORKS:
        1. Takes a minority sample (fraud case)
        2. Finds its K nearest neighbors (other fraud cases)
        3. Creates a NEW synthetic sample between them
        4. Repeats until minority class is balanced

        """
        print(f"   Handling class imbalance with SMOTE...")
        print(f"   Before: {pd.Series(y_train).value_counts().to_dict()}")

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"   After:  {pd.Series(y_resampled).value_counts().to_dict()}")

        return X_resampled, y_resampled

    def scale_features(self, X_train, X_test):
        """
        StandardScaler: transforms features to mean=0, std=1.

        - Some features are in dollars (amount_usd: 0-1,000,000)
        - Others are 0/1 flags (is_weekend: 0 or 1)
        - Scaling makes all features comparable

        IMPORTANT: fit on train only, transform both.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)   # Learn from train
        X_test_scaled = self.scaler.transform(X_test)          # Apply same to test

        return X_train_scaled, X_test_scaled

    def prepare_all(self):
        """Run the full preparation pipeline"""
        df = self.load_features()
        X, y = self.select_and_validate(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train, y_train = self.handle_imbalance(X_train, y_train)
        X_train, X_test = self.scale_features(X_train, X_test)
        return X_train, X_test, y_train, y_test

# MODEL TRAINING AND EVALUATION

class ModelTrainer:
    """
    Trains models and logs everything to MLflow 
    (Logging Hyperparameter, metrics, models artifacts,plots)
    (run visually at http://localhost:5000/)
    """

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.results = []

    def _evaluate(self, y_true, y_pred, y_prob):
        """
        Calculate all metrics
        Returns a dictionary
        """
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(pr_recall, pr_precision)
        
        return {
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': pr_auc,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
        }
    
    def _log_plots(self, y_true, y_prob, model_name):
        """
        Generate and log evaluation plots to MLflow.

        ROC Curve:  True Positive Rate vs False Positive Rate
        PR Curve:   Precision vs Recall (better for imbalanced data)
        Confusion Matrix: Actual vs Predicted counts
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc_val:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()

        # --- PR Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(recall, precision)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')

        # --- Confusion Matrix ---
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        im = axes[2].imshow(cm, cmap='Blues')
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        axes[2].set_xticklabels(['Normal', 'Fraud'])
        axes[2].set_yticklabels(['Normal', 'Fraud'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)

        plt.suptitle(model_name, fontsize=14, fontweight='bold')
        plt.tight_layout()

        mlflow.log_figure(fig, f"{model_name}_evaluation.png")
        plt.close()

    def _log_feature_importance(self, importances, model_name):
        """Plot and log top 15 most important features"""
        feat_imp = pd.Series(importances, index=FEATURE_COLUMNS)
        top15 = feat_imp.nlargest(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        top15.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top 15 Feature Importances')
        plt.tight_layout()

        mlflow.log_figure(fig, f"{model_name}_feature_importance.png")
        plt.close()

# Individual Model Trainers

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """
        Random Forest: An ensemble of decision trees.
        
        - Good baseline model
        - Handles imbalanced data with class_weight='balanced'
        - Feature importances are reliable
        - Less prone to overfitting than single trees
        """
        print(" Training Random Forest...")

        params = {
            'n_estimators': 200,       # 200 trees
            'max_depth': 15,           # Tree depth limit (prevents overfitting)
            'min_samples_split': 10,   # Min samples to split a node
            'min_samples_leaf': 4,     # Min samples in leaf node
            'class_weight': 'balanced', # Auto-adjusts for imbalance
            'random_state': 42,
            'n_jobs': -1               # Use all CPU cores
        }

        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_params(params)

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            metrics = self._evaluate(y_test, y_pred, y_prob)
            mlflow.log_metrics(metrics)

            self._log_plots(y_test, y_prob, 'Random Forest')
            self._log_feature_importance(model.feature_importances_, 'Random Forest')

            mlflow.sklearn.log_model(model, "model")

            self.results.append(('Random Forest', metrics, model))
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """
        XGBoost: Gradient Boosted Decision Trees.

        - Usually top performer on tabular data
        - scale_pos_weight handles imbalance (like class_weight for boosting)
        - Very fast training with GPU support
        - Built-in regularization prevents overfitting
        """
        print("Training XGBoost...")

        # scale_pos_weight = ratio of negative to positive samples
        # Tells the model to penalize missing a fraud case more
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,      # How much each tree contributes
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,          # Use 80% of data per tree
            'colsample_bytree': 0.8,   # Use 80% of features per tree
            'min_child_weight': 5,
            'random_state': 42,
            'eval_metric': 'auc',
            'use_label_encoder': False
        }

        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params(params)

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            metrics = self._evaluate(y_test, y_pred, y_prob)
            mlflow.log_metrics(metrics)

            self._log_plots(y_test, y_prob, 'XGBoost')
            self._log_feature_importance(model.feature_importances_, 'XGBoost')

            mlflow.xgboost.log_model(model, "model")

            self.results.append(('XGBoost', metrics, model))
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        """
        LightGBM: Light Gradient Boosting Machine.

        WHY use it:
        - Fastest of the three models
        - Handles large datasets very well
        - Often wins Kaggle competitions
        - is_unbalance=True handles class imbalance natively
        """
        print("\n🚀 Training LightGBM...")

        params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,          # Controls tree complexity
            'is_unbalance': True,      # Auto-handles imbalanced data
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1              # Suppress training output
        }

        with mlflow.start_run(run_name="lightgbm"):
            mlflow.log_params(params)

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            metrics = self._evaluate(y_test, y_pred, y_prob)
            mlflow.log_metrics(metrics)

            self._log_plots(y_test, y_prob, 'LightGBM')
            self._log_feature_importance(model.feature_importances_, 'LightGBM')

            mlflow.lightgbm.log_model(model, "model")

            self.results.append(('LightGBM', metrics, model))
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

# Compare and save best

    def compare_and_save_best(self, scaler):
        """
        Compare all models and save the best one.

        - It balances Precision and Recall equally
        - Perfect for fraud detection where both matter
        - Accuracy would be misleading
        """
        print(" MODEL COMPARISON")
        print(f"{'Model':<20} {'ROC-AUC':<10} {'PR-AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")

        for name, metrics, _ in self.results:
            print(f"{name:<20} {metrics['roc_auc']:<10.4f} {metrics['pr_auc']:<10.4f} "
                  f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")

        # Pick best by F1
        best_name, best_metrics, best_model = max(self.results, key=lambda x: x[1]['f1_score'])

        print(f"Best model: {best_name} (F1={best_metrics['f1_score']:.4f})")

        # Save best model + scaler
        models_dir = project_root / 'models'
        models_dir.mkdir(exist_ok=True)

        joblib.dump(best_model, models_dir / 'best_model.joblib')
        joblib.dump(scaler, models_dir / 'scaler.joblib')

        # Save metadata
        metadata = {
            'model_name': best_name,
            'metrics': best_metrics,
            'trained_at': datetime.now().isoformat(),
            'feature_columns': FEATURE_COLUMNS
        }

        import json
        with open(models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f" Saved to models/")
        print(f"   - best_model.joblib")
        print(f"   - scaler.joblib")
        print(f"   - model_metadata.json")

        return best_model, best_metrics

# ENTRY POINT


if __name__ == "__main__":
    print(" AML FRAUD DETECTION - MODEL TRAINING")

    # 1. Prepare data
    preparer = DataPreparer()
    X_train, X_test, y_train, y_test = preparer.prepare_all()

    # 2. Train all models
    trainer = ModelTrainer()
    trainer.train_random_forest(X_train, X_test, y_train, y_test)
    trainer.train_xgboost(X_train, X_test, y_train, y_test)
    trainer.train_lightgbm(X_train, X_test, y_train, y_test)

    # 3. Compare and save
    best_model, best_metrics = trainer.compare_and_save_best(preparer.scaler)

    print(" Training complete! MLflow at http://localhost:5000")