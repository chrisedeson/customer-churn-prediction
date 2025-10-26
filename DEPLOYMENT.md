# Deployment Guide

## Streamlit Community Cloud Deployment

### Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (sign up at share.streamlit.io)

### Steps

#### 1. Push to GitHub
```bash
# Create a new repository on GitHub (e.g., customer-churn-prediction)
# Then push your local repository:

git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/customer-churn-prediction`
4. Set main file path: `app.py`
5. Click "Deploy"

#### 3. Important Notes

**Data Files:**
- Processed data files (CSV) in `data/processed/` are ignored by git
- Models in `models/` are ignored by git
- On first deployment, run data preparation and training:
  - The app will show an error with instructions
  - Run `make data` and `make train` locally
  - Commit and push the generated files (temporarily remove from .gitignore)

**Alternative: Run Pipeline on Deployment**
Add a `.streamlit/config.toml` file with startup script, or modify app.py to run pipeline on first load.

**Recommended Approach for Deployment:**
1. Temporarily comment out lines in .gitignore for data and models:
```bash
# Comment these lines in .gitignore:
# *.csv
# *.pkl
```

2. Add processed files and models:
```bash
git add data/processed/*.csv models/*.pkl
git commit -m "Add processed data and trained models for deployment"
git push
```

3. Deploy to Streamlit Cloud

4. After deployment, uncomment .gitignore lines and commit

### Performance Optimization

**Memory Management:**
- Streamlit Cloud has 1GB RAM limit
- Use @st.cache_resource for models (loads once)
- Use @st.cache_data for data (cached efficiently)

**Speed Optimization:**
- Precomputed models (no training on load)
- Limit SHAP samples to 200 for global explanations
- Single prediction should be <2 seconds

### Environment Variables (if needed)
Set in Streamlit Cloud dashboard under "Advanced settings":
```
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

### Troubleshooting

**Out of Memory Error:**
- Reduce SHAP sample size in interpretability.py
- Use feature importance fallback instead of SHAP

**Slow Loading:**
- Ensure @st.cache_resource is used for models
- Verify models are pre-trained (not training on load)

**Module Not Found:**
- Check all imports use `from src.` prefix
- Verify requirements.txt has all dependencies

### Post-Deployment
1. Test the live app thoroughly
2. Update README.md with public URL
3. Create video demo showing the deployed app
4. Submit project with both repository and app URLs
