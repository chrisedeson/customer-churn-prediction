"""
Data preparation pipeline for customer churn prediction.
Handles loading, cleaning, feature engineering, and train/val/test splitting.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import urllib.request

from config import (
    DATA_URL, RAW_DATA_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_STATE, TARGET_COL, DROP_COLS,
    TENURE_BUCKETS, SERVICE_COLUMNS, DATA_RAW, DATA_PROCESSED
)


def download_data():
    """Download raw data from IBM repository."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_FILE.exists():
        print(f"Data already exists at {RAW_DATA_FILE}")
        return
    
    print(f"Downloading data from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_FILE)
    print(f"Data saved to {RAW_DATA_FILE}")


def load_raw_data():
    """Load raw data from CSV."""
    if not RAW_DATA_FILE.exists():
        download_data()
    
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def clean_data(df):
    """Clean data and handle missing values."""
    df = df.copy()
    
    # Handle TotalCharges: convert to numeric and fill missing with tenure * monthly
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # For customers with tenure=0 or missing TotalCharges, use MonthlyCharges
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges']
    
    print(f"Filled {missing_mask.sum()} missing TotalCharges values")
    
    # Convert binary target to numeric
    df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0})
    
    # Drop identifier columns
    df = df.drop(columns=DROP_COLS, errors='ignore')
    
    return df


def create_tenure_bucket(tenure):
    """Categorize tenure into buckets."""
    for bucket, (min_val, max_val) in TENURE_BUCKETS.items():
        if min_val <= tenure < max_val:
            return bucket
    return "24m+"


def engineer_features(df):
    """Create business-driven features."""
    df = df.copy()
    
    # Tenure buckets
    df['tenure_bucket'] = df['tenure'].apply(create_tenure_bucket)
    
    # Services count: count non-'No' services
    services_count = 0
    for col in SERVICE_COLUMNS:
        if col in df.columns:
            # Count as 1 if not 'No' and not 'No phone service' and not 'No internet service'
            services_count += (~df[col].isin(['No', 'No phone service', 'No internet service'])).astype(int)
    
    df['services_count'] = services_count
    
    # Monthly to total ratio (handle division by zero)
    expected_total = df['tenure'] * df['MonthlyCharges']
    expected_total = expected_total.replace(0, 1)  # Avoid division by zero
    df['monthly_to_total_ratio'] = df['TotalCharges'] / expected_total
    
    # Internet but no tech support flag
    has_internet = df['InternetService'].isin(['DSL', 'Fiber optic'])
    no_tech_support = df['TechSupport'] == 'No'
    df['internet_no_tech_support'] = (has_internet & no_tech_support).astype(int)
    
    # Fiber optic with no security
    is_fiber = df['InternetService'] == 'Fiber optic'
    no_security = df['OnlineSecurity'] == 'No'
    df['fiber_no_security'] = (is_fiber & no_security).astype(int)
    
    # Senior with month-to-month
    is_senior = df['SeniorCitizen'] == 1
    is_mtm = df['Contract'] == 'Month-to-month'
    df['senior_mtm'] = (is_senior & is_mtm).astype(int)
    
    return df


def encode_categorical(df):
    """Encode categorical variables using LabelEncoder (alphabetical sorting)."""
    df = df.copy()
    
    # Identify categorical columns (excluding target and engineered numerics)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Store encoders for later use
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    return df, encoders


def split_data(df):
    """Split data into train/val/test with stratification."""
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO)),
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL]
    )
    
    # Second split: val vs test
    val_size = VAL_RATIO / (VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO))
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COL]
    )
    
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df):
    """Save processed splits to CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    print(f"Saved train to {TRAIN_FILE}")
    print(f"Saved val to {VAL_FILE}")
    print(f"Saved test to {TEST_FILE}")


def prepare_data():
    """Main pipeline to prepare data."""
    print("=" * 50)
    print("Starting data preparation pipeline")
    print("=" * 50)
    
    # Load and clean
    df = load_raw_data()
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categorical
    df, encoders = encode_categorical(df)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df)
    
    print("=" * 50)
    print("Data preparation complete")
    print("=" * 50)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_data()
