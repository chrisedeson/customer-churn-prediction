"""
Interpretability module for model explanations.
Provides SHAP explanations for tree models and coefficient analysis for linear models.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.config import (
    LOGISTIC_MODEL, RF_MODEL, XGB_MODEL, PREPROCESSOR_FILE,
    TEST_FILE, TARGET_COL
)


def load_models():
    """Load trained models and preprocessor."""
    logistic = joblib.load(LOGISTIC_MODEL)
    rf = joblib.load(RF_MODEL)
    xgb = joblib.load(XGB_MODEL)
    scaler = joblib.load(PREPROCESSOR_FILE)
    return logistic, rf, xgb, scaler


def get_feature_importance_tree(model, feature_names):
    """
    Get feature importance for tree-based models.
    Returns DataFrame sorted by importance.
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def get_feature_importance_logistic(model, feature_names, scaler=None):
    """
    Get feature importance for Logistic Regression using standardized coefficients.
    Formula: importance = |coefficient × std_dev_of_feature|
    
    Since features are already scaled by StandardScaler, we use absolute coefficients.
    """
    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'importance': np.abs(coefficients)
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df, title="Feature Importance", top_n=15):
    """Plot feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def get_shap_explainer(model, model_type='tree'):
    """
    Get SHAP explainer for a model.
    Use TreeExplainer for tree-based models.
    """
    try:
        import shap
        
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            # For linear models, use LinearExplainer or skip SHAP
            explainer = None
        
        return explainer
    except ImportError:
        print("SHAP not available. Using fallback feature importance.")
        return None


def compute_shap_values(explainer, X, max_samples=200):
    """
    Compute SHAP values for given samples.
    Limit samples to avoid performance issues.
    """
    if explainer is None:
        return None
    
    # Sample data if too large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X
    
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, title="SHAP Summary"):
    """Create SHAP summary plot."""
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(title)
        plt.tight_layout()
        return fig
    except ImportError:
        print("SHAP not available for summary plot")
        return None


def explain_local_prediction(model, X_input, feature_names, model_type='tree'):
    """
    Explain a single prediction using SHAP or feature importance.
    Returns explanation data for visualization.
    """
    try:
        import shap
        
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Get base value and prediction
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            # Create explanation DataFrame
            explanation_df = pd.DataFrame({
                'feature': feature_names,
                'value': X_input.iloc[0].values,
                'shap_value': shap_values[0]
            }).sort_values('shap_value', key=abs, ascending=False)
            
            return explanation_df, base_value
        else:
            # For logistic regression, use coefficients
            coefficients = model.coef_[0]
            contribution = X_input.iloc[0].values * coefficients
            
            explanation_df = pd.DataFrame({
                'feature': feature_names,
                'value': X_input.iloc[0].values,
                'contribution': contribution
            }).sort_values('contribution', key=abs, ascending=False)
            
            return explanation_df, None
    
    except ImportError:
        # Fallback: use feature importance
        if model_type == 'tree':
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_[0])
        
        contribution = X_input.iloc[0].values * importance
        
        explanation_df = pd.DataFrame({
            'feature': feature_names,
            'value': X_input.iloc[0].values,
            'contribution': contribution
        }).sort_values('contribution', key=abs, ascending=False)
        
        return explanation_df, None


def plot_local_explanation(explanation_df, top_n=10, title="Prediction Explanation"):
    """Plot local prediction explanation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = explanation_df.head(top_n)
    
    # Use SHAP values if available, otherwise use contribution
    value_col = 'shap_value' if 'shap_value' in top_features.columns else 'contribution'
    values = top_features[value_col].values
    
    colors = ['red' if v < 0 else 'green' for v in values]
    
    ax.barh(range(len(top_features)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Impact on Prediction')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_all_explanations():
    """Generate explanations for all models on test data."""
    print("=" * 50)
    print("Generating model explanations")
    print("=" * 50)
    
    # Load models and data
    logistic, rf, xgb, scaler = load_models()
    test_df = pd.read_csv(TEST_FILE)
    
    X_test = test_df.drop(columns=[TARGET_COL])
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    feature_names = X_test.columns.tolist()
    
    # Logistic Regression importance
    print("\nLogistic Regression Feature Importance:")
    lr_importance = get_feature_importance_logistic(logistic, feature_names)
    print(lr_importance.head(10))
    
    # Random Forest importance
    print("\nRandom Forest Feature Importance:")
    rf_importance = get_feature_importance_tree(rf, feature_names)
    print(rf_importance.head(10))
    
    # XGBoost importance
    print("\nXGBoost Feature Importance:")
    xgb_importance = get_feature_importance_tree(xgb, feature_names)
    print(xgb_importance.head(10))
    
    print("\n" + "=" * 50)
    print("Explanations generated")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_explanations()
