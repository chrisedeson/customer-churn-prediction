"""Quick test of app loading functionality."""
import sys
sys.path.insert(0, '.')

from app import load_models, load_processed_data

print("Testing app components...")
print("\n1. Loading data...")
train, val, test = load_processed_data()
print(f"   ✓ Data loaded: {len(train)} train, {len(val)} val, {len(test)} test samples")

print("\n2. Loading models...")
models = load_models()
print(f"   ✓ Models loaded: {', '.join(models.keys())}")

print("\n3. Testing prediction...")
from src.predict import make_prediction
sample = test.drop(columns=['Churn']).head(1)
result = make_prediction(sample, models, model_choice='xgb')
print(f"   ✓ Prediction successful: {result['churn_percentage']:.1f}% churn risk")

print("\n✅ All app components working correctly!")
