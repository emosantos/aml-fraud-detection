# AML Fraud Detection Pipeline

Status: Work in Progress An end-to-end data engineering and machine learning system designed to detect money laundering patterns in large-scale financial datasets.
## Overview

This project implements a Medallion Architecture to process and analyze over 5 million transactions from the [IBM AML dataset](https://ibm.ent.box.com/v/AML-Anti-Money-Laundering-Data). It transitions data from raw ingestion to refined feature sets for anomaly detection.

## Data Strategy

    Bronze : Raw ingestion of CSV data into PostgreSQL with batch tracking.

    Silver : Data cleaning, deduplication, and schema standardization.

    Gold : Feature engineering and ML model training

## Project Structure

aml-fraud-detection/
├── airflow/       # Pipeline orchestration (DAGs)
├── config/        # Database and system configurations
├── data/          # Local data storage (Raw, Processed, Features)
├── docker/        # PostgreSQL container setup
├── notebooks/     # EDA and model prototyping
├── src/           # Core logic (Ingestion, Processing, Models)


## 🛠️ Tech Stack

- **Database**: PostgreSQL
- **Data Processing**: Pandas, PySpark
- **ML**: Scikit-learn, XGBoost, LightGBM, SMOTE
- **MLOps**: MLflow (experiment tracking)
- **API**: FastAPI
- **Orchestration**: Apache Airflow (planned)
- **Visualization**: Streamlit, Plotly
- **Tooling**: Docker
