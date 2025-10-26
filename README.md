# Customer Churn Prediction & CLV

Predicts customer churn probability and estimates Customer Lifetime Value (CLV) for SaaS retention strategy.

GitHub: https://github.com/chrisedeson/customer-churn-prediction

## Quick Start

```bash
./setup.sh          # Setup virtual environment and install dependencies
source venv/bin/activate
make data           # Download and prepare data
make train          # Train models
make run            # Start Streamlit app
```

## Deployment

Live app: [URL will be added after deployment]

## Models

- Logistic Regression (baseline)
- Random Forest  
- XGBoost

Performance: AUC-ROC 0.80-0.85, Recall 60%+

## CLV Assumptions

Expected Tenure = 24 months for non-churners, 6 months for churners
CLV = MonthlyCharges × ExpectedTenure

## Project Structure

```
src/
  config.py           # Configuration and constants
  data_prep.py        # Data loading and feature engineering
  clv_analysis.py     # CLV computation and quartile analysis
  train_models.py     # Model training and evaluation
  interpretability.py # SHAP and feature importance
  predict.py          # Prediction utilities
app.py                # Streamlit application
Makefile              # Automation commands
```

## Requirements

Python 3.9+. See requirements.txt for dependencies.

## Video Demo

[Link will be added]
