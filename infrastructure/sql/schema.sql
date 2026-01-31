-- DB Schema

-- Bronze Layer: Raw Data

CREATE SCHEMA IF NOT EXISTS bronze_layer;

CREATE TABLE bronze_layer.raw_transaction (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    from_bank VARCHAR(100),
    from_account VARCHAR(100),
    to_bank VARCHAR(100),
    to_account VARCHAR(100),
    amount_received DECIMAL(18,2),
    receiving_currency VARCHAR(10),
    amount_paid DECIMAL(18,2),
    payment_currency VARCHAR(10),
    payment_format VARCHAR(50),
    is_laundering BOOLEAN,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    batch_id VARCHAR(100)
    );

-- Silver Layer: Clean data

CREATE SCHEMA IF NOT EXISTS silver_layer;

CREATE TABLE silver_layer.processed_transactions (
    transaction_id SERIAL PRIMARY KEY,
    transaction_date DATE NOT NULL,
    transaction_hour INTEGER,
    from_bank VARCHAR(100),
    from_account VARCHAR(100),
    to_bank VARCHAR(100),
    to_account VARCHAR(100),
    amount_usd DECIMAL(18,2),
    payment_format VARCHAR(50),
    is_cross_border BOOLEAN,
    currency_mismatch BOOLEAN,
    is_laundering BOOLEAN,
    processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

CREATE INDEX idx_proc_date ON silver_layer.processed_transactions(transaction_date);
CREATE INDEX idx_proc_account ON silver_layer.processed_transactions(from_account);

-- Gold Layer: Features & Predictions

CREATE SCHEMA IF NOT EXISTS gold_layer;

CREATE TABLE gold_layer.fraud_predictions (
    prediction_id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES silver_layer.processed_transactions(transaction_id),
    model_version VARCHAR(50),
    fraud_probability DECIMAL(7,6),
    fraud_prediction BOOLEAN,
    risk_level VARCHAR(20),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE gold_layer.model_performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    evaluation_date DATE,
    roc_auc DECIMAL(7,6),
    precision_score DECIMAL(7,6),
    recall_score DECIMAL(7,6),
    f1_score DECIMAL(7,6),
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);