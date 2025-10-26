# Customer Churn Prediction & CLV

Predicts customer churn probability and estimates Customer Lifetime Value (CLV) for SaaS retention strategy.

## Quick Start

```bash
make setup
make data
make train
make run
```

## Deployment

Live app: [URL will be added after deployment]

## Models

- Logistic Regression (baseline)
- Random Forest
- XGBoost

## CLV Assumptions

Expected Tenure = 24 months for non-churners, 6 months for churners.
CLV = MonthlyCharges × ExpectedTenure

## Structure

```
src/
  config.py         # Configuration and constants
  data_prep.py      # Data loading and feature engineering
  clv_analysis.py   # CLV computation and quartile analysis
  train_models.py   # Model training and evaluation
  interpretability.py # SHAP and feature importance
  predict.py        # Prediction utilities
app.py              # Streamlit application
```

## Requirements

Python 3.9+. See requirements.txt for dependencies.

## Video Demo

[Link will be added]
