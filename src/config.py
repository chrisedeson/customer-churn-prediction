"""
Configuration and constants for the churn prediction project.
Centralizes paths, model parameters, and business assumptions.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Data source
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_FILE = DATA_RAW / "Telco-Customer-Churn.csv"

# Processed data files
TRAIN_FILE = DATA_PROCESSED / "train.csv"
VAL_FILE = DATA_PROCESSED / "val.csv"
TEST_FILE = DATA_PROCESSED / "test.csv"

# Model files
LOGISTIC_MODEL = MODELS_DIR / "logistic.pkl"
RF_MODEL = MODELS_DIR / "rf.pkl"
XGB_MODEL = MODELS_DIR / "xgb.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"

# Data split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_STATE = 42

# CLV assumptions (in months)
EXPECTED_TENURE_NON_CHURNER = 24
EXPECTED_TENURE_CHURNER = 6

# Feature engineering
TENURE_BUCKETS = {
    "0-6m": (0, 6),
    "6-12m": (6, 12),
    "12-24m": (12, 24),
    "24m+": (24, float('inf'))
}

# Services to count
SERVICE_COLUMNS = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

# Model hyperparameters
LOGISTIC_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'solver': 'liblinear'
}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_leaf': 4,
    'min_samples_split': 10,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'random_state': RANDOM_STATE,
    'scale_pos_weight': 3,  # Handle class imbalance
    'eval_metric': 'logloss',
    'n_jobs': -1
}

# Target column
TARGET_COL = 'Churn'

# Columns to drop (identifier, leakage, etc.)
DROP_COLS = ['customerID']

# Risk thresholds
RISK_LOW = 0.33
RISK_MEDIUM = 0.66
