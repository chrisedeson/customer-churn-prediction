"""
Prediction utilities for making churn predictions and computing CLV.
Provides functions for single predictions and batch predictions.
"""
import pandas as pd
import numpy as np
import joblib

from src.config import (
    LOGISTIC_MODEL, RF_MODEL, XGB_MODEL, PREPROCESSOR_FILE,
    EXPECTED_TENURE_NON_CHURNER, EXPECTED_TENURE_CHURNER,
    RISK_LOW, RISK_MEDIUM
)


def load_prediction_artifacts():
    """Load all models and preprocessor for prediction."""
    logistic = joblib.load(LOGISTIC_MODEL)
    rf = joblib.load(RF_MODEL)
    xgb = joblib.load(XGB_MODEL)
    scaler = joblib.load(PREPROCESSOR_FILE)
    
    return {
        'logistic': logistic,
        'rf': rf,
        'xgb': xgb,
        'scaler': scaler
    }


def preprocess_input(input_df, scaler):
    """Preprocess input features using fitted scaler."""
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
    return input_scaled_df


def predict_churn(model, X_input):
    """
    Predict churn probability for input.
    Returns probability of churn (0-1).
    """
    proba = model.predict_proba(X_input)[:, 1]
    return proba


def predict_churn_ensemble(models_dict, X_input):
    """
    Predict churn using ensemble of all models.
    Returns average probability across models.
    """
    probas = []
    
    for model_name in ['logistic', 'rf', 'xgb']:
        model = models_dict[model_name]
        proba = predict_churn(model, X_input)
        probas.append(proba)
    
    ensemble_proba = np.mean(probas, axis=0)
    return ensemble_proba


def get_risk_label(churn_probability):
    """
    Categorize churn probability into risk levels.
    Returns: 'Low Risk', 'Medium Risk', or 'High Risk'
    """
    if churn_probability < RISK_LOW:
        return "Low Risk"
    elif churn_probability < RISK_MEDIUM:
        return "Medium Risk"
    else:
        return "High Risk"


def compute_clv_for_prediction(monthly_charges, churn_probability):
    """
    Compute CLV for a single prediction.
    Uses weighted expected tenure based on churn probability.
    """
    expected_tenure = (
        churn_probability * EXPECTED_TENURE_CHURNER +
        (1 - churn_probability) * EXPECTED_TENURE_NON_CHURNER
    )
    clv = monthly_charges * expected_tenure
    return clv, expected_tenure


def make_prediction(input_data, models_dict, model_choice='xgb', use_ensemble=False):
    """
    Make a complete prediction including churn probability, risk, and CLV.
    
    Args:
        input_data: DataFrame with features
        models_dict: Dictionary with models and scaler
        model_choice: Which model to use ('logistic', 'rf', 'xgb')
        use_ensemble: Whether to use ensemble prediction
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess input
    X_input = preprocess_input(input_data, models_dict['scaler'])
    
    # Get churn probability
    if use_ensemble:
        churn_proba = predict_churn_ensemble(models_dict, X_input)[0]
        model_used = "Ensemble"
    else:
        model = models_dict[model_choice]
        churn_proba = predict_churn(model, X_input)[0]
        model_used = model_choice.upper()
    
    # Get risk label
    risk_label = get_risk_label(churn_proba)
    
    # Compute CLV (need monthly charges from original input)
    monthly_charges = input_data['MonthlyCharges'].iloc[0]
    clv, expected_tenure = compute_clv_for_prediction(monthly_charges, churn_proba)
    
    result = {
        'churn_probability': churn_proba,
        'churn_percentage': churn_proba * 100,
        'risk_label': risk_label,
        'clv': clv,
        'expected_tenure': expected_tenure,
        'monthly_charges': monthly_charges,
        'model_used': model_used
    }
    
    return result


def format_prediction_output(result):
    """Format prediction result for display."""
    output = f"""
Churn Prediction Results
{'=' * 50}
Model Used: {result['model_used']}
Churn Probability: {result['churn_percentage']:.1f}%
Risk Level: {result['risk_label']}

Customer Lifetime Value
{'=' * 50}
Monthly Charges: ${result['monthly_charges']:.2f}
Expected Tenure: {result['expected_tenure']:.1f} months
Estimated CLV: ${result['clv']:.2f}
"""
    return output


def batch_predict(input_df, models_dict, model_choice='xgb'):
    """
    Make predictions for multiple customers.
    Returns DataFrame with predictions.
    """
    # Preprocess all inputs
    X_input = preprocess_input(input_df, models_dict['scaler'])
    
    # Get predictions
    model = models_dict[model_choice]
    churn_probas = predict_churn(model, X_input)
    
    # Create results DataFrame
    results_df = input_df.copy()
    results_df['churn_probability'] = churn_probas
    results_df['churn_percentage'] = churn_probas * 100
    results_df['risk_label'] = [get_risk_label(p) for p in churn_probas]
    
    # Compute CLV for each
    clvs = []
    expected_tenures = []
    for i, prob in enumerate(churn_probas):
        monthly = input_df.iloc[i]['MonthlyCharges']
        clv, tenure = compute_clv_for_prediction(monthly, prob)
        clvs.append(clv)
        expected_tenures.append(tenure)
    
    results_df['clv'] = clvs
    results_df['expected_tenure'] = expected_tenures
    
    return results_df


if __name__ == "__main__":
    # Example usage
    print("Loading models...")
    models = load_prediction_artifacts()
    print("Models loaded successfully")
    
    # Test with sample data
    from src.config import TEST_FILE, TARGET_COL
    test_df = pd.read_csv(TEST_FILE)
    sample = test_df.drop(columns=[TARGET_COL]).head(1)
    
    result = make_prediction(sample, models, model_choice='xgb')
    print(format_prediction_output(result))
