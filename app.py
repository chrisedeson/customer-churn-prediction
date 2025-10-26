"""
Customer Churn Prediction & CLV Streamlit Application.
Single-page app with tabs for Predict, Model Performance, and CLV Overview.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import (
    LOGISTIC_MODEL, RF_MODEL, XGB_MODEL, PREPROCESSOR_FILE,
    TRAIN_FILE, VAL_FILE, TEST_FILE, TARGET_COL
)
from predict import (
    load_prediction_artifacts, make_prediction,
    preprocess_input, predict_churn
)
from interpretability import (
    get_feature_importance_tree, get_feature_importance_logistic,
    plot_feature_importance, explain_local_prediction, plot_local_explanation
)
from clv_analysis import (
    compute_clv, create_clv_quartiles, compute_churn_rate_by_quartile
)
from train_models import evaluate_model


# Page config
st.set_page_config(
    page_title="Churn Prediction & CLV",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_processed_data():
    """Load processed train, val, and test data."""
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, val_df, test_df


@st.cache_resource
def load_models():
    """Load all trained models and preprocessor."""
    return load_prediction_artifacts()


@st.cache_data
def compute_test_metrics(_models_dict, test_df):
    """Compute metrics for all models on test data."""
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # Preprocess
    X_test_scaled = pd.DataFrame(
        _models_dict['scaler'].transform(X_test),
        columns=X_test.columns
    )
    
    # Evaluate each model
    all_metrics = []
    predictions = {}
    
    for model_name in ['logistic', 'rf', 'xgb']:
        model = _models_dict[model_name]
        display_name = {
            'logistic': 'Logistic Regression',
            'rf': 'Random Forest',
            'xgb': 'XGBoost'
        }[model_name]
        
        metrics, y_pred, y_pred_proba = evaluate_model(
            model, X_test_scaled, y_test, display_name
        )
        all_metrics.append(metrics)
        predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    df_metrics = pd.DataFrame(all_metrics)
    return df_metrics, predictions, y_test


def create_input_form():
    """Create input form for prediction."""
    st.subheader("Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=50.0)
    
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    
    with col3:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
    
    col4, col5 = st.columns(2)
    
    with col4:
        streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    with col5:
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Bank transfer (automatic)", "Credit card (automatic)", 
             "Electronic check", "Mailed check"]
        )
    
    return {
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }


def engineer_features_from_input(input_dict):
    """Apply same feature engineering as training data."""
    df = pd.DataFrame([input_dict])
    
    # Tenure bucket
    tenure = df['tenure'].iloc[0]
    if tenure < 6:
        df['tenure_bucket'] = '0-6m'
    elif tenure < 12:
        df['tenure_bucket'] = '6-12m'
    elif tenure < 24:
        df['tenure_bucket'] = '12-24m'
    else:
        df['tenure_bucket'] = '24m+'
    
    # Services count
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    services_count = 0
    for col in service_cols:
        if df[col].iloc[0] not in ['No', 'No phone service', 'No internet service']:
            services_count += 1
    df['services_count'] = services_count
    
    # Monthly to total ratio
    expected_total = max(1, tenure * df['MonthlyCharges'].iloc[0])
    df['monthly_to_total_ratio'] = df['TotalCharges'].iloc[0] / expected_total
    
    # Internet but no tech support
    has_internet = df['InternetService'].iloc[0] in ['DSL', 'Fiber optic']
    no_tech = df['TechSupport'].iloc[0] == 'No'
    df['internet_no_tech_support'] = 1 if (has_internet and no_tech) else 0
    
    # Fiber no security
    is_fiber = df['InternetService'].iloc[0] == 'Fiber optic'
    no_security = df['OnlineSecurity'].iloc[0] == 'No'
    df['fiber_no_security'] = 1 if (is_fiber and no_security) else 0
    
    # Senior with month-to-month
    is_senior = df['SeniorCitizen'].iloc[0] == 1
    is_mtm = df['Contract'].iloc[0] == 'Month-to-month'
    df['senior_mtm'] = 1 if (is_senior and is_mtm) else 0
    
    return df


def encode_input_features(df, train_df):
    """Encode categorical features to match training data encoding."""
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on train data to get same encoding
        le.fit(train_df[col])
        df[col] = le.transform(df[col])
    
    # Ensure same column order as training
    df = df[train_df.columns]
    
    return df


def render_predict_tab(models_dict, train_df):
    """Render the Predict tab."""
    st.header("Churn Prediction")
    
    input_dict = create_input_form()
    
    if st.button("Predict Churn", type="primary"):
        # Engineer features
        input_df = engineer_features_from_input(input_dict)
        
        # Encode features
        X_input = encode_input_features(input_df, train_df.drop(columns=[TARGET_COL]))
        
        # Make prediction
        result = make_prediction(X_input, models_dict, model_choice='xgb')
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{result['churn_percentage']:.1f}%")
        with col2:
            st.metric("Risk Level", result['risk_label'])
        with col3:
            st.metric("Estimated CLV", f"${result['clv']:.2f}")
        
        # Show CLV breakdown
        st.subheader("CLV Calculation")
        st.info(f"""
        **Formula:** CLV = Monthly Charges × Expected Tenure
        
        - Monthly Charges: ${result['monthly_charges']:.2f}
        - Expected Tenure: {result['expected_tenure']:.1f} months
        - **Estimated CLV: ${result['clv']:.2f}**
        
        Note: Expected tenure is computed as a weighted average based on churn probability 
        (6 months for churners, 24 months for non-churners).
        """)
        
        # Local explanation
        st.subheader("Prediction Explanation")
        
        try:
            model = models_dict['xgb']
            X_scaled = preprocess_input(X_input, models_dict['scaler'])
            explanation_df, _ = explain_local_prediction(
                model, X_scaled, X_input.columns.tolist(), model_type='tree'
            )
            
            fig = plot_local_explanation(explanation_df, top_n=10)
            st.pyplot(fig)
            plt.close()
            
            st.caption("Green bars increase churn risk, red bars decrease churn risk.")
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")


def render_model_performance_tab(models_dict, test_df):
    """Render the Model Performance tab."""
    st.header("Model Performance")
    
    # Compute metrics
    df_metrics, predictions, y_test = compute_test_metrics(models_dict, test_df)
    
    # Display metrics table
    st.subheader("Test Set Metrics")
    st.dataframe(
        df_metrics.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'AUC-ROC': '{:.4f}'
        }),
        use_container_width=True
    )
    
    # Model selection for detailed view
    st.subheader("Detailed Analysis")
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    model_key = {
        "Logistic Regression": "logistic",
        "Random Forest": "rf",
        "XGBoost": "xgb"
    }[model_choice]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        
        y_pred = predictions[model_key]['y_pred']
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        cm = confusion_matrix(y_test_array, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Churn', 'Churn'])
        ax.set_yticklabels(['No Churn', 'Churn'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=20)
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # ROC curves
        st.subheader("ROC Curves")
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        
        for name, key in [("Logistic", "logistic"), ("Random Forest", "rf"), ("XGBoost", "xgb")]:
            y_pred_proba = predictions[key]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test_array, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    # Feature importance
    st.subheader("Feature Importance")
    
    model = models_dict[model_key]
    X_test = test_df.drop(columns=[TARGET_COL])
    feature_names = X_test.columns.tolist()
    
    if model_key == 'logistic':
        importance_df = get_feature_importance_logistic(model, feature_names)
    else:
        importance_df = get_feature_importance_tree(model, feature_names)
    
    fig = plot_feature_importance(importance_df, title=f"{model_choice} Feature Importance")
    st.pyplot(fig)
    plt.close()


def render_clv_overview_tab(train_df):
    """Render the CLV Overview tab."""
    st.header("Customer Lifetime Value Overview")
    
    # Compute CLV
    train_with_clv = compute_clv(train_df)
    train_with_clv = create_clv_quartiles(train_with_clv)
    
    # CLV statistics
    st.subheader("CLV Distribution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean CLV", f"${train_with_clv['clv'].mean():.2f}")
    with col2:
        st.metric("Median CLV", f"${train_with_clv['clv'].median():.2f}")
    with col3:
        st.metric("Min CLV", f"${train_with_clv['clv'].min():.2f}")
    with col4:
        st.metric("Max CLV", f"${train_with_clv['clv'].max():.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CLV Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(train_with_clv['clv'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Customer Lifetime Value ($)')
        ax.set_ylabel('Count')
        ax.set_title('CLV Distribution')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Churn Rate by CLV Quartile")
        churn_by_quartile = compute_churn_rate_by_quartile(train_with_clv)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        quartiles = churn_by_quartile.index
        churn_rates = churn_by_quartile['churn_rate'] * 100
        
        bars = ax.bar(quartiles, churn_rates, edgecolor='black', alpha=0.7)
        ax.set_xlabel('CLV Quartile')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title('Churn Rate by CLV Quartile')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Insights
    st.subheader("Key Insights")
    
    high_premium_churn = churn_by_quartile.loc[['High', 'Premium'], 'churn_rate'].mean()
    low_med_churn = churn_by_quartile.loc[['Low', 'Med'], 'churn_rate'].mean()
    premium_churn = churn_by_quartile.loc['Premium', 'churn_rate']
    premium_total = churn_by_quartile.loc['Premium', 'total']
    
    st.info(f"""
    **Retention Strategy Recommendations:**
    
    1. High-value customers (High/Premium quartiles) show {high_premium_churn:.1%} churn rate 
       compared to {low_med_churn:.1%} for lower-value segments, indicating significant revenue at risk.
    
    2. The Premium segment comprises {premium_total} customers with {premium_churn:.1%} churn rate, 
       representing the highest priority for retention investments.
    
    3. Focus retention programs on Premium and High CLV segments to maximize ROI. 
       These customers generate the most revenue and warrant proactive engagement to prevent churn.
    """)
    
    # Detailed quartile breakdown
    st.subheader("Quartile Breakdown")
    st.dataframe(
        churn_by_quartile.style.format({
            'total': '{:,.0f}',
            'churned': '{:,.0f}',
            'churn_rate': '{:.2%}'
        }),
        use_container_width=True
    )


def main():
    """Main application entry point."""
    st.title("Customer Churn Prediction & CLV Analysis")
    
    # Load data and models
    try:
        train_df, val_df, test_df = load_processed_data()
        models_dict = load_models()
    except FileNotFoundError as e:
        st.error(f"""
        Required files not found. Please run the data preparation and model training pipeline first:
        
        ```bash
        make data
        make train
        ```
        
        Error: {str(e)}
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Predict", "Model Performance", "CLV Overview"])
    
    with tab1:
        render_predict_tab(models_dict, train_df)
    
    with tab2:
        render_model_performance_tab(models_dict, test_df)
    
    with tab3:
        render_clv_overview_tab(train_df)


if __name__ == "__main__":
    main()
