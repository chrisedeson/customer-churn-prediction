"""Test high-risk customer scenario from task requirements."""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.predict import load_prediction_artifacts, make_prediction
from src.config import TRAIN_FILE, TARGET_COL

# Load train data for encoding reference
train_df = pd.read_csv(TRAIN_FILE)

# High-risk customer profile from task:
# Senior citizen, month-to-month, fiber optic, no security/backup/tech support,
# electronic check, monthly charges >= $100
high_risk_customer = {
    'SeniorCitizen': 1,
    'tenure': 2,  # Short tenure
    'MonthlyCharges': 105.0,
    'TotalCharges': 210.0,
    'gender': 'Male',
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check'
}

# Load models
print("Loading models...")
models = load_prediction_artifacts()

# Prepare input
input_df = pd.DataFrame([high_risk_customer])

# Engineer features (manually since we're outside the app)
tenure = input_df['tenure'].iloc[0]
if tenure < 6:
    input_df['tenure_bucket'] = '0-6m'
elif tenure < 12:
    input_df['tenure_bucket'] = '6-12m'
elif tenure < 24:
    input_df['tenure_bucket'] = '12-24m'
else:
    input_df['tenure_bucket'] = '24m+'

# Services count
service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']
services_count = 0
for col in service_cols:
    if input_df[col].iloc[0] not in ['No', 'No phone service', 'No internet service']:
        services_count += 1
input_df['services_count'] = services_count

# Monthly to total ratio
expected_total = max(1, tenure * input_df['MonthlyCharges'].iloc[0])
input_df['monthly_to_total_ratio'] = input_df['TotalCharges'].iloc[0] / expected_total

# Flags
has_internet = input_df['InternetService'].iloc[0] in ['DSL', 'Fiber optic']
no_tech = input_df['TechSupport'].iloc[0] == 'No'
input_df['internet_no_tech_support'] = 1 if (has_internet and no_tech) else 0

is_fiber = input_df['InternetService'].iloc[0] == 'Fiber optic'
no_security = input_df['OnlineSecurity'].iloc[0] == 'No'
input_df['fiber_no_security'] = 1 if (is_fiber and no_security) else 0

is_senior = input_df['SeniorCitizen'].iloc[0] == 1
is_mtm = input_df['Contract'].iloc[0] == 'Month-to-month'
input_df['senior_mtm'] = 1 if (is_senior and is_mtm) else 0

# Encode
from sklearn.preprocessing import LabelEncoder
categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train_df[col])
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        # Handle unseen labels
        input_df[col] = le.transform([train_df[col].mode()[0]])[0]

# Ensure same column order
input_df = input_df[train_df.drop(columns=[TARGET_COL]).columns]

# Make prediction
result = make_prediction(input_df, models, model_choice='xgb')

print("\n" + "="*60)
print("HIGH-RISK CUSTOMER TEST")
print("="*60)
print(f"Churn Probability: {result['churn_percentage']:.1f}%")
print(f"Risk Level: {result['risk_label']}")
print(f"Expected: >60% churn probability")
print()
if result['churn_percentage'] > 60:
    print("✓ TEST PASSED - High risk customer correctly identified")
else:
    print("✗ TEST FAILED - Expected >60% churn probability")
print("="*60)
