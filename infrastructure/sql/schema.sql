-- DB Schema
-- Run Automatically when Docker is created

-- UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Bronze Layer: Raw Data

CREATE SCHEMA IF NOT EXISTS bronze_layer;

CREATE TABLE IF NOT EXISTS bronze_layer.raw_transactions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    from_bank VARCHAR(100),
    from_account VARCHAR(100),
    to_bank VARCHAR(100),
    to_account VARCHAR(100),
    amount_received DECIMAL(18,2),
    receiving_currency VARCHAR(100),
    amount_paid DECIMAL(18,2),
    payment_currency VARCHAR(100),
    payment_format VARCHAR(100),
    is_laundering BOOLEAN,
    batch_id VARCHAR(100),
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON bronze_layer.raw_transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_from_account ON bronze_layer.raw_transactions(from_account);
CREATE INDEX IF NOT EXISTS idx_raw_to_account ON bronze_layer.raw_transactions(to_account);

-- Raw accounts from CSV

CREATE TABLE IF NOT EXISTS bronze_layer.raw_accounts (
    id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100),
    bank_id VARCHAR(50),
    account_number VARCHAR(100),
    entity_id VARCHAR(100),
    entity_name VARCHAR(200),
    batch_id VARCHAR(100),
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_accounts_account ON bronze_layer.raw_accounts(account_number);
CREATE INDEX IF NOT EXISTS idx_raw_accounts_entity ON bronze_layer.raw_accounts(entity_id);
CREATE INDEX IF NOT EXISTS idx_raw_accounts_bank ON bronze_layer.raw_accounts(bank_id);

-- Silver Layer: Clean data

CREATE SCHEMA IF NOT EXISTS silver_layer;

CREATE TABLE IF NOT EXISTS silver_layer.processed_transactions (
    transaction_id SERIAL PRIMARY KEY,
    transaction_date DATE NOT NULL,
    transaction_hour INTEGER,
    from_bank VARCHAR(100),
    from_account VARCHAR(100),
    to_bank VARCHAR(100),
    to_account VARCHAR(100),
    amount_usd DECIMAL(18,2),
    original_amount DECIMAL(18,2),
    original_currency VARCHAR(50),
    payment_format VARCHAR(100),
    is_cross_border BOOLEAN,
    currency_mismatch BOOLEAN,
    is_laundering BOOLEAN,
    processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_proc_date ON silver_layer.processed_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_proc_from_account ON silver_layer.processed_transactions(from_account);
CREATE INDEX IF NOT EXISTS idx_proc_to_account ON silver_layer.processed_transactions(to_account);
CREATE INDEX IF NOT EXISTS idx_proc_amount ON silver_layer.processed_transactions(amount_usd);

-- Cleaned accounts
-- Enriched with transaction counts and basic stats
CREATE TABLE IF NOT EXISTS silver_layer.processed_accounts (
    account_id VARCHAR(100) PRIMARY KEY,
    bank_name VARCHAR(100),
    bank_id VARCHAR(50),
    entity_id VARCHAR(100),
    entity_name VARCHAR(200),
    total_transactions INTEGER DEFAULT 0,
    total_volume_usd DECIMAL(18,2) DEFAULT 0,
    processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_proc_accounts_entity ON silver_layer.processed_accounts(entity_id);
CREATE INDEX IF NOT EXISTS idx_proc_accounts_bank ON silver_layer.processed_accounts(bank_id);

-- Gold Layer: Features & Predictions

CREATE SCHEMA IF NOT EXISTS gold_layer;

CREATE TABLE IF NOT EXISTS gold_layer.fraud_predictions (
    prediction_id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES silver_layer.processed_transactions(transaction_id),
    model_version VARCHAR(50),
    fraud_probability DECIMAL(7,6),
    fraud_prediction BOOLEAN,
    risk_level VARCHAR(20) CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_txn ON gold_layer.fraud_predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON gold_layer.fraud_predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_prob ON gold_layer.fraud_predictions(fraud_probability DESC);

CREATE TABLE IF NOT EXISTS gold_layer.model_performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    evaluation_date DATE,
    total_predictions INTEGER,
    roc_auc DECIMAL(7,6),
    precision_score DECIMAL(7,6),
    recall_score DECIMAL(7,6),
    f1_score DECIMAL(7,6),
    true_positives INTEGER,
    false_positives INTEGER,
    true_negatives INTEGER,
    false_negatives INTEGER,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_date ON gold_layer.model_performance_metrics(evaluation_date DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_version ON gold_layer.model_performance_metrics(model_version);

-- Data drift monitoring (tracks if data distribution changes over time)
CREATE TABLE IF NOT EXISTS gold_layer.data_drift_metrics (
    drift_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    evaluation_date DATE,
    ks_statistic DECIMAL(7,6),
    p_value DECIMAL(10,8),
    drift_detected BOOLEAN,
    reference_mean DECIMAL(18,6),
    current_mean DECIMAL(18,6),
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_drift_date ON gold_layer.data_drift_metrics(evaluation_date DESC);
CREATE INDEX IF NOT EXISTS idx_drift_detected ON gold_layer.data_drift_metrics(drift_detected);

CREATE SCHEMA IF NOT EXISTS audit;

-- Tracks every pipeline run (ingestion, processing, training)
-- Useful for monitoring
CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    run_id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED')),
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    rows_processed INTEGER,
    rows_failed INTEGER,
    error_message TEXT,
    batch_id VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_pipeline_status ON audit.pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_started ON audit.pipeline_runs(started_at DESC);