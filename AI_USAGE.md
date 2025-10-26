# AI Usage Documentation

## Overview

AI (GitHub Copilot) was used throughout this project to accelerate development while maintaining code quality and best practices. This document details how AI was utilized and what was verified or modified.

## Project Setup and Structure

**AI Assistance:**
- Generated initial project directory structure following best practices
- Created modular architecture with clear separation of concerns (config, data prep, training, prediction, interpretability)
- Designed configuration module to centralize constants and avoid code duplication

**Manual Verification:**
- Validated directory structure matches task requirements
- Confirmed all paths and configuration values are correct
- Ensured scalability through modular design

## Data Preparation (src/data_prep.py)

**AI Assistance:**
- Generated data loading and cleaning logic
- Implemented feature engineering functions (tenure buckets, services count, ratios, flags)
- Created stratified train/val/test split logic

**Manual Verification:**
- Verified TotalCharges missing value handling (filled with MonthlyCharges for tenure=0)
- Confirmed LabelEncoder produces alphabetical sorting as required
- Validated feature engineering matches task specifications
- Tested split ratios (60/20/20) with stratification on target variable

## CLV Analysis (src/clv_analysis.py)

**AI Assistance:**
- Implemented CLV computation formula (MonthlyCharges × ExpectedTenure)
- Created quartile segmentation and churn rate analysis
- Generated visualization functions

**Manual Verification:**
- Confirmed CLV assumptions (24 months non-churners, 6 months churners) match task
- Validated quartile labels (Low/Med/High/Premium)
- Reviewed business insights generation logic

## Model Training (src/train_models.py)

**AI Assistance:**
- Implemented training pipeline for all three models (Logistic Regression, Random Forest, XGBoost)
- Generated evaluation metrics computation
- Created model persistence logic

**Manual Verification:**
- Tuned hyperparameters for each model based on task guidelines:
  - Added class_weight='balanced' for Logistic Regression
  - Set scale_pos_weight=3 for XGBoost to handle class imbalance
  - Configured max_depth and min_samples for Random Forest
- Verified train+val combination for final training as per task
- Confirmed StandardScaler is fit only on training data
- Validated all metrics (Precision, Recall, F1, AUC-ROC) are computed correctly

## Interpretability (src/interpretability.py)

**AI Assistance:**
- Implemented SHAP TreeExplainer for tree-based models
- Created feature importance extraction for all model types
- Generated visualization functions for global and local explanations

**Manual Verification:**
- Verified fallback to coefficient analysis for Logistic Regression (faster and more appropriate)
- Confirmed SHAP values are extracted correctly for binary classification
- Tested sample limiting (200 samples) to avoid performance issues
- Validated local explanation handles both SHAP and fallback methods

## Prediction Module (src/predict.py)

**AI Assistance:**
- Created prediction pipeline with preprocessing
- Implemented risk categorization logic
- Generated CLV computation for predictions using weighted expected tenure

**Manual Verification:**
- Confirmed risk thresholds (33%/66%) are appropriate
- Validated CLV formula uses probability-weighted tenure
- Tested batch prediction functionality
- Verified ensemble prediction averaging logic

## Streamlit App (app.py)

**AI Assistance:**
- Generated three-tab layout structure
- Created input form with all required features
- Implemented prediction display and visualizations

**Manual Verification:**
- Simplified UI to avoid excessive styling (professional, minimal)
- Verified feature engineering in app matches training pipeline
- Confirmed encoding matches training data (alphabetical LabelEncoder)
- Tested caching decorators (@st.cache_data, @st.cache_resource)
- Validated all visualizations render correctly
- Ensured error handling for missing files

## Critical Prompts and Decisions

**Key Prompt 1: "Make it scalable, DRY, industry standard"**
- Result: Created config.py to centralize all constants and hyperparameters
- Result: Modular functions with single responsibility
- Result: Avoided code duplication across modules

**Key Prompt 2: "Handle class imbalance"**
- Result: Added class_weight='balanced' to Logistic Regression
- Result: Set scale_pos_weight=3 for XGBoost
- Result: Added balanced class weight to Random Forest

**Key Prompt 3: "LabelEncoder alphabetical encoding"**
- Result: Documented that LabelEncoder sorts alphabetically
- Result: Ensured app encoding matches training encoding
- Result: Gender: Female=0, Male=1

## What AI Generated vs. What Was Modified

**Generated Without Modification:**
- File structure and imports
- Basic data loading and splitting logic
- Visualization functions
- Makefile structure

**Generated With Modifications:**
- Model hyperparameters (tuned for class imbalance)
- CLV assumptions (verified 24/6 months)
- Risk thresholds (validated 33%/66% splits)
- Feature engineering logic (added all required features)
- App input form (ensured all fields match training)

## Testing and Validation

**Manual Testing:**
- Ran full pipeline: data prep → training → prediction
- Verified high-risk customer test case (senior, month-to-month, fiber optic, no support services)
- Confirmed model performance meets requirements (Recall ≥ 60%, AUC ≥ 0.80)
- Tested Streamlit app with various inputs
- Validated feature importance shows Contract, Tenure, MonthlyCharges as top features

**What AI Could Not Do:**
- Determine appropriate hyperparameter values (required domain knowledge)
- Validate business logic (CLV assumptions, risk thresholds)
- Test edge cases and error conditions
- Optimize for specific performance requirements
- Make architectural decisions about tradeoffs

## Conclusion

AI significantly accelerated development by generating boilerplate code, implementing standard algorithms, and creating visualizations. However, all business logic, hyperparameters, and critical decisions were manually verified and adjusted. The final code represents a collaboration where AI provided structure and AI provided domain expertise and validation.
