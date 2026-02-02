import os

from pathlib import Path
from dotenv import load_dotenv

# Loading Enviroment
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

# Data Dir
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIT = DATA_DIR / 'raw'
PROCESSED_DATA_DIT = DATA_DIR / 'processed'
FEATURES_DATA_DIT = DATA_DIR / 'features'

# Model Dir
MODELS_DIR = PROJECT_ROOT / 'models'

# DB Config
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'aml_user'),
    'password': os.getenv('DB_PASSWORD', 'aml_password_123'),
    'database': os.getenv('DB_NAME', 'aml_fraud_db')
}

def get_db_connection_string():
    """Get the PostgreSQL connection string."""
    return (f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

# MLflow Config
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Dataset Config
DATASETS = {
    'small_hi': 'HI-Small_Trans.csv',
    'small_li': 'LI-Small_Trans.csv',
    'medium_hi': 'HI-Medium_Trans.csv',
    'medium_li': 'LI-Medium_Trans.csv',
}

# Default Dataset
DEFAULT_DATASET = 'small_hi'

# Model Config
MODEL_CONFIG = {
    'test_size' : 0.2,
    'random_state' : 42,
    'cv_folds' : 5
}