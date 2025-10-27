"""
Customer Churn Prediction & CLV Streamlit Application.
Single-page app with tabs for Predict, Model Performance, and CLV Overview.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container */
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Compact form spacing */
    .stSelectbox, .stNumberInput {
        margin-bottom: -15px !important;
    }
    
    div[data-testid="stExpander"] > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric cards with gradient backgrounds */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Expanders - more compact */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.05rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 8px;
        padding: 0.5rem 1rem !important;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #4a5568;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Dividers */
    hr {
        margin: 1rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
    }
    </style>
    """, unsafe_allow_html=True)


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
    # Load only essential models - lazy load RF only when needed
    import joblib
    from src.config import LOGISTIC_MODEL, XGB_MODEL, PREPROCESSOR_FILE
    
    models = {
        'logistic': joblib.load(LOGISTIC_MODEL),
        'xgb': joblib.load(XGB_MODEL),
        'scaler': joblib.load(PREPROCESSOR_FILE)
    }
    return models


@st.cache_resource
def load_rf_model():
    """Load Random Forest model only when needed (it's 12MB)."""
    import joblib
    from src.config import RF_MODEL
    return joblib.load(RF_MODEL)


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
    
    # Load RF model only when needed (lazy loading)
    if 'rf' not in _models_dict:
        _models_dict['rf'] = load_rf_model()
    
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
    
    # Demographics section
    with st.expander("👤 Demographics & Account", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col3:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        
        with col4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    # Services section
    with st.expander("📞 Services", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col2:
            online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        
        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
    
    # Billing section
    with st.expander("💳 Billing", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        
        with col2:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=50.0)
        
        with col3:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
        with col4:
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


@st.cache_data(show_spinner=False)
def engineer_features_from_input(input_dict):
    """Apply same feature engineering as training data. Cached for performance."""
    df = pd.DataFrame([input_dict])
    
    # Tenure bucket - encoded numerically (alphabetical: 0-6m=0, 12-24m=1, 24m+=2, 6-12m=3)
    tenure = df['tenure'].iloc[0]
    if tenure < 6:
        df['tenure_bucket'] = 0  # '0-6m'
    elif tenure < 12:
        df['tenure_bucket'] = 3  # '6-12m'
    elif tenure < 24:
        df['tenure_bucket'] = 1  # '12-24m'
    else:
        df['tenure_bucket'] = 2  # '24m+'
    
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


@st.cache_data(show_spinner=False)
def encode_input_features(df, train_df):
    """Encode categorical features to match training data encoding (alphabetical). Cached for performance."""
    df = df.copy()
    
    # Manual encoding maps (alphabetical order as used in data_prep.py)
    encoding_maps = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 
                         'Electronic check': 2, 'Mailed check': 3}
    }
    
    # Apply encoding
    for col, mapping in encoding_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Ensure same column order as training
    df = df[train_df.columns]
    
    return df


def render_predict_tab(models_dict, train_df):
    """Render the Predict tab."""
    st.header("Customer Churn Prediction")
    st.markdown("Enter customer information below to predict churn probability and estimated lifetime value.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state for prediction results
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    input_dict = create_input_form()
    
    # Center the button with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)
    
    if predict_btn:
        # Engineer features
        input_df = engineer_features_from_input(input_dict)
        
        # Encode features
        X_input = encode_input_features(input_df, train_df.drop(columns=[TARGET_COL]))
        
        # Make prediction (fast - no spinner needed)
        st.session_state.prediction_result = make_prediction(X_input, models_dict, model_choice='xgb')
    
    # Display results if prediction exists
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        
        # Display results with enhanced visuals
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            # Beautiful gauge chart for churn probability
            churn_prob = result['churn_percentage']
            
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            # Create gauge
            import matplotlib.patches as mpatches
            
            # Background arc (full semicircle)
            theta = np.linspace(0, np.pi, 100)
            r = 1.0
            
            # Color segments for gauge background
            colors_bg = ['#10b981', '#fbbf24', '#ef4444']  # green, yellow, red
            segments = [(0, 33), (33, 66), (66, 100)]
            
            for i, (start, end) in enumerate(segments):
                theta_seg = np.linspace(np.pi * (100-end)/100, np.pi * (100-start)/100, 50)
                x = r * np.cos(theta_seg)
                y = r * np.sin(theta_seg)
                ax.fill_between(x, 0, y, color=colors_bg[i], alpha=0.3)
            
            # Needle
            angle = np.pi * (100 - churn_prob) / 100
            needle_length = 0.8
            ax.plot([0, needle_length * np.cos(angle)], [0, needle_length * np.sin(angle)], 
                   'k-', linewidth=4, zorder=10)
            ax.plot([0], [0], 'ko', markersize=12, zorder=11)
            
            # Add percentage text in center
            ax.text(0, -0.3, f'{churn_prob:.1f}%', 
                   ha='center', va='center', fontsize=32, fontweight='bold',
                   color='#1f2937')
            ax.text(0, -0.5, 'Churn Probability', 
                   ha='center', va='center', fontsize=11, color='#6b7280')
            
            # Add scale labels
            ax.text(-0.9, 0.1, '0', ha='center', fontsize=10, color='#6b7280')
            ax.text(0, 1.1, '50', ha='center', fontsize=10, color='#6b7280')
            ax.text(0.9, 0.1, '100', ha='center', fontsize=10, color='#6b7280')
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.6, 1.3)
            ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        with col2:
            # Risk assessment box
            risk_label = result['risk_label']
            risk_colors = {
                "Low": "#10b981",
                "Medium": "#fbbf24", 
                "High": "#ef4444"
            }
            risk_color = risk_colors.get(risk_label, "#6b7280")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}11 100%); 
                        border-left: 5px solid {risk_color}; 
                        padding: 1.5rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem;">
                <h3 style="color: #1f2937; margin: 0 0 0.5rem 0; font-size: 1.3rem;">
                    🎯 Risk Level
                </h3>
                <div style="background: {risk_color}; 
                           color: white; 
                           padding: 0.75rem 1.5rem; 
                           border-radius: 8px; 
                           font-size: 1.2rem; 
                           font-weight: 700;
                           text-align: center;">
                    {risk_label} Risk
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # CLV metrics row
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_clv = 733.46  # Average from training data
            clv_diff = result['clv'] - avg_clv
            st.metric(
                "💰 Estimated CLV", 
                f"${result['clv']:.2f}",
                delta=f"${clv_diff:+.2f} vs avg",
                delta_color="normal" if clv_diff > 0 else "inverse"
            )
        
        with col2:
            st.metric(
                "💵 Monthly Charges", 
                f"${result['monthly_charges']:.2f}"
            )
        
        with col3:
            st.metric(
                "📅 Expected Tenure", 
                f"{result['expected_tenure']:.1f} months"
            )
        
        # Show CLV breakdown
        with st.expander("💡 CLV Calculation Details", expanded=False):
            st.markdown(f"""
            **Formula:** CLV = Monthly Charges × Expected Tenure
            
            - **Monthly Charges:** ${result['monthly_charges']:.2f}
            - **Expected Tenure:** {result['expected_tenure']:.1f} months
            - **Estimated CLV:** ${result['clv']:.2f}
            
            *Expected tenure is computed as a weighted average based on churn probability 
            (6 months for churners, 24 months for non-churners).*
            """)
        
        # Local explanation (optional - behind expander to avoid slow loading)
        st.divider()
        
        with st.expander("🔍 Feature Impact Analysis (Click to view)", expanded=False):
            try:
                with st.spinner("Generating explanation..."):
                    model = models_dict['xgb']
                    X_scaled = preprocess_input(X_input, models_dict['scaler'])
                    explanation_df, _ = explain_local_prediction(
                        model, X_scaled, X_input.columns.tolist(), model_type='tree'
                    )
                    
                    fig = plot_local_explanation(explanation_df, top_n=10)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    
                    st.caption("🟢 Green bars increase churn risk | 🔴 Red bars decrease churn risk")
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
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Use beautiful gradient colors
        cmap = LinearSegmentedColormap.from_list('custom', ['#f0f9ff', '#0284c7', '#0c4a6e'])
        im = ax.imshow(cm, cmap=cmap, alpha=0.9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Churn', 'Churn'], fontsize=12, fontweight='600')
        ax.set_yticklabels(['No Churn', 'Churn'], fontsize=12, fontweight='600')
        ax.set_xlabel('Predicted', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_ylabel('Actual', fontsize=13, fontweight='600', color='#1f2937')
        
        # Add text annotations with better styling
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                             color="white" if cm[i, j] > cm.max() / 2 else "#1f2937", 
                             fontsize=22, fontweight='700')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        # ROC curves
        st.subheader("ROC Curves")
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        
        colors = {
            'Logistic': '#8b5cf6',  # Purple
            'Random Forest': '#f59e0b',  # Orange
            'XGBoost': '#10b981'  # Green
        }
        for name, key in [("Logistic", "logistic"), ("Random Forest", "rf"), ("XGBoost", "xgb")]:
            y_pred_proba = predictions[key]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test_array, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
                   linewidth=3, color=colors[name], alpha=0.9)
        
        ax.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', alpha=0.6, linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='700', pad=15, color='#1f2937')
        ax.legend(loc='lower right', frameon=True, fontsize=11, framealpha=0.95, edgecolor='#e5e7eb')
        ax.grid(alpha=0.15, linestyle='--', color='#94a3b8')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
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
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Beautiful gradient colors for histogram
        n, bins, patches = ax.hist(train_with_clv['clv'], bins=50, edgecolor='white', 
                                    alpha=0.9, linewidth=0.8)
        
        # Apply gradient coloring to bars
        cm = plt.colormaps.get_cmap('viridis')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        ax.set_xlabel('Customer Lifetime Value ($)', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_ylabel('Frequency', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_title('CLV Distribution', fontsize=14, fontweight='700', pad=15, color='#1f2937')
        ax.grid(axis='y', alpha=0.15, linestyle='--', color='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.subheader("Churn Rate by CLV Quartile")
        churn_by_quartile = compute_churn_rate_by_quartile(train_with_clv)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        quartiles = churn_by_quartile.index
        churn_rates = churn_by_quartile['churn_rate'] * 100
        
        # Beautiful gradient colors
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        bars = ax.bar(quartiles, churn_rates, color=colors, edgecolor='white', 
                     alpha=0.9, linewidth=1.5)
        
        # Add gradient effect to bars
        for bar, color in zip(bars, colors):
            bar.set_edgecolor('white')
            bar.set_linewidth(1.5)
        
        ax.set_xlabel('CLV Quartile', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_ylabel('Churn Rate (%)', fontsize=13, fontweight='600', color='#1f2937')
        ax.set_title('Churn Rate by CLV Quartile', fontsize=14, fontweight='700', pad=15, color='#1f2937')
        ax.grid(axis='y', alpha=0.15, linestyle='--', color='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='700', color='#1f2937')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
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
    # Header
    st.title("📊 Customer Churn Prediction & CLV Analysis")
    st.markdown("""
    <div style='padding: 1rem 0; margin-bottom: 1.5rem;'>
        <p style='font-size: 1.1rem; opacity: 0.8; margin: 0;'>
            Predict customer churn probability, analyze model performance, and explore customer lifetime value insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    try:
        train_df, val_df, test_df = load_processed_data()
        models_dict = load_models()
    except FileNotFoundError as e:
        st.error(f"""
        **Required files not found.** Please run the data preparation and model training pipeline first:
        
        ```bash
        make data
        make train
        ```
        
        Error: {str(e)}
        """)
        return
    
    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application provides:
        
        **🎯 Churn Prediction**
        - Individual customer risk assessment
        - Multiple ML models (Logistic, RF, XGBoost)
        - Feature importance explanations
        
        **📈 Model Performance**
        - Comprehensive metrics comparison
        - ROC curves and confusion matrices
        - Feature importance analysis
        
        **💰 CLV Analysis**
        - Customer lifetime value insights
        - Quartile-based segmentation
        - Retention strategy recommendations
        """)
        
        st.divider()
        
        # Dataset info
        st.subheader("Dataset Info")
        st.metric("Total Customers", f"{len(train_df) + len(val_df) + len(test_df):,}")
        st.metric("Training Samples", f"{len(train_df):,}")
        st.metric("Features", len(train_df.columns) - 1)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📈 Model Performance", "💰 CLV Overview"])
    
    with tab1:
        render_predict_tab(models_dict, train_df)
    
    with tab2:
        render_model_performance_tab(models_dict, test_df)
    
    with tab3:
        render_clv_overview_tab(train_df)


if __name__ == "__main__":
    main()
