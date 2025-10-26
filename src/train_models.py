"""
Model training pipeline for churn prediction.
Trains Logistic Regression, Random Forest, and XGBoost with light tuning.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

from src.config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, TARGET_COL,
    LOGISTIC_MODEL, RF_MODEL, XGB_MODEL, PREPROCESSOR_FILE,
    LOGISTIC_PARAMS, RF_PARAMS, XGB_PARAMS, MODELS_DIR
)


def load_data():
    """Load train, validation, and test sets."""
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df):
    """Separate features and target, fit scaler on training data."""
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(**LOGISTIC_PARAMS)
    model.fit(X_train, y_train)
    print("Logistic Regression trained")
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    print("Random Forest trained")
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    print("\nTraining XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    print("XGBoost trained")
    return model


def evaluate_model(model, X, y, model_name="Model"):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba


def print_metrics(metrics):
    """Print model metrics in a formatted way."""
    print(f"\n{metrics['Model']} Metrics:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    print(f"  AUC-ROC:   {metrics['AUC-ROC']:.4f}")


def save_models(logistic, rf, xgb, scaler):
    """Save trained models and preprocessor."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(logistic, LOGISTIC_MODEL)
    joblib.dump(rf, RF_MODEL)
    joblib.dump(xgb, XGB_MODEL)
    joblib.dump(scaler, PREPROCESSOR_FILE)
    
    print(f"\nSaved models to {MODELS_DIR}/")


def create_evaluation_report(all_metrics):
    """Create a DataFrame with all model metrics."""
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.round(4)
    return df_metrics


def train_all_models():
    """Main pipeline to train all models."""
    print("=" * 50)
    print("Starting model training pipeline")
    print("=" * 50)
    
    # Load data
    train_df, val_df, test_df = load_data()
    print(f"Loaded train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")
    
    # Prepare features (scaled)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_features(
        train_df, val_df, test_df
    )
    
    # Combine train and val for final training (as per task requirements)
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    print(f"\nCombined train+val: {len(X_train_full)} samples")
    
    # Train models
    logistic = train_logistic_regression(X_train_full, y_train_full)
    rf = train_random_forest(X_train_full, y_train_full)
    xgb = train_xgboost(X_train_full, y_train_full)
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set")
    print("=" * 50)
    
    all_metrics = []
    
    metrics_lr, _, _ = evaluate_model(logistic, X_test, y_test, "Logistic Regression")
    print_metrics(metrics_lr)
    all_metrics.append(metrics_lr)
    
    metrics_rf, _, _ = evaluate_model(rf, X_test, y_test, "Random Forest")
    print_metrics(metrics_rf)
    all_metrics.append(metrics_rf)
    
    metrics_xgb, _, _ = evaluate_model(xgb, X_test, y_test, "XGBoost")
    print_metrics(metrics_xgb)
    all_metrics.append(metrics_xgb)
    
    # Create evaluation report
    df_metrics = create_evaluation_report(all_metrics)
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(df_metrics.to_string(index=False))
    
    # Save models
    save_models(logistic, rf, xgb, scaler)
    
    print("\n" + "=" * 50)
    print("Model training complete")
    print("=" * 50)
    
    return logistic, rf, xgb, scaler, df_metrics


if __name__ == "__main__":
    train_all_models()
