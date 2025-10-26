# **Project 2: Customer Churn Prediction & Customer Lifetime Value (CLV)**

## **Business Context**

SaaS companies lose 5–7% of revenue annually to churn. This project shows you can (1) predict who is likely to churn and (2) estimate which customers are most valuable to retain using **Customer Lifetime Value (CLV)**—all in a deployed, interactive app.

---

## **What You’ll Build (Scope)**

* **Data prep** with a few business-driven features

* **CLV analysis** (assumptions documented, segments compared)

* **Three models:** Logistic Regression, Random Forest, XGBoost

* **Interpretability:** SHAP explanations **or** feature-importance fallback

* **Single-page Streamlit app** (tabs for Predict, Model Performance, CLV)

* **Cloud deployment** (Streamlit Community Cloud)

* **Short video demo (required)**

---

## **Deliverables**

### **1\) Data Preparation**

* Load & clean the IBM **Telco Customer Churn** dataset.

* Handle `TotalCharges` missing values (state your approach).

* Engineer features (keep it simple and explainable):

  * `tenure_bucket`: 0–6m, 6–12m, 12–24m, 24m+

  * `services_count`: total number of services

  * `monthly_to_total_ratio`: `TotalCharges / max(1, tenure * MonthlyCharges)`

  * flags like “internet but no tech support”

* Define **Expected Tenure** (state your assumption clearly).

* **CLV \= MonthlyCharges × ExpectedTenure (months)**.

* **Split:** 60/20/20 train/val/test with **stratification**.

* Save processed splits to `data/processed/`.

**1a) Data Preparation Tips**

* When encoding categorical variables, remember LabelEncoder sorts alphabetically:   
  * Gender: Female=0, Male=1 (not Male=0, Female=1)  
  * PaymentMethod will be alphabetically sorted  
  * MultipleLines: No=0, No Phone Service=1, Yes=2  
  * Your app’s prediction encoding must match this\!

### **2\) CLV Analysis**

* Compute CLV for each customer.

* Split into **quartiles** (“Low/Med/High/Premium”).

* Report **churn rate by CLV quartile** (1–2 clear charts)

* Write **2–3 business insights** (short bullets).- include these in your app / video

### **3\) Modeling (3 models)**

* Train:

  * Logistic Regression (baseline)

  * Random Forest

  * XGBoost

* Do light tuning (2–3 useful hyperparameters/model).

* **Evaluate on test**: Precision, Recall, F1, **AUC-ROC**.

* Persist trained models \+ preprocessors in `models/`.

**3a) Model Accuracy Expectations and Improvement Tips:**

* **Baseline Expectations:**  
  * Initial models: 80-38% AUC-ROC is good  
  * After improvement: 34-36% AUC-ROC is achievable  
  * Industry Benchmark: 80-90% for churn prediction  
* **Key improvements to try:** (ask AI what these mean)  
  * Handle class imbalance  
  * Feature engineering \- add custom features just like we did for recur-scan  
  * Hyperparameter tips:  
    * XGBoost: max\_depth and learning\_rate  
    * Random Forest: max\_depth and min\_samples\_leaf  
    * Use train+val together for final model training  
* **Common pitfalls to avoid**  
  * RF and XGB should both beat Logistic Regression  
  * If models show 40% recall, you have a class imbalance issue  
* **Be sure to test:**  
  * A senior citizen with a month-to-month contract, fiber-optic internet, no security/backup/tech support services, paying by electronic check and monthly charges \>= $100 should be at high risk of churn (\>60%).  
  * Contract, Tenure, and MonthlyCharges should have high feature importances  
  * Recall should be at least 60%

### **4\) Interpretability (SHAP or Fallback)**

* **For Tree-based models (RF/XGB):**  Use SHAP TreeExplainer  
  * Global: Summary or bar plot (top features)  
  * Local: single-customer explanation in the app  
* **For Logistic Regression:** Use coefficient analysis  
  * Display absolute standardized coefficients as feature importance  
  * Formula: importance \= |coefficient \* std\_dev\_of\_feature|  
  * This is faster and more appropriate for linear models  
  * Skip SHAP’s KernelExplainer for Logistic Regression \- it’s too slow and coefficients are more interpretable anyway  
* **Tip:**  if SHAP is slow, **sample** (e.g., 100-200 rows) for global plots

### **5\) Streamlit App (Single Page)**

Use tabs (e.g., `st.tabs(["Predict", "Model Performance", "CLV Overview"])`).

**Tab 1 – Predict**

* Input form for key features (friendly labels; validate types).

* Output:

  * Churn probability (0–100%) \+ Low/Med/High label

  * **Local explanation**: SHAP plot *or* fallback feature-importance for this input

  * Estimated **CLV** for this customer (show formula snippet)

**Tab 2 – Model Performance**

* Metrics table (Precision, Recall, F1, AUC) for all 3 models.

* ROC curves (overlay) and confusion matrix for selected model.

* **Global importance**: SHAP summary *or* fallback feature importance.

**Tab 3 – CLV Overview**

* CLV distribution (histogram or bars).

* Churn rate by CLV quartile.

* A one-paragraph takeaway: who to prioritize and why.

### **6\) Deployment**

* Deploy on **Streamlit Community Cloud**.

* App should load quickly; single prediction should be **\< 2 seconds**.

* Provide:

  * **Public URL**

  * `requirements.txt` with pinned versions (see below)

  * `README.md` (business framing, CLV assumptions, how to run)

  * `AI_USAGE.md` (what AI helped with; the prompts that mattered; what you fixed/verified)

**7\) Optional enhancements**

* Ensemble prediction: average all 3 models for final prediction  
* Feature engineering: add interaction terms (e.g., senior \+ fiber\_optic) as new features so logistic regression can use them better

---

## **Performance & Caching (Important)**

Use Streamlit caching to keep the app snappy:

**Cache data** (processed splits):  
@st.cache\_data  
def load\_processed():  
    ...  
    return df\_train, df\_val, df\_test

**Cache models & explainers** (heavy objects):  
@st.cache\_resource  
def load\_models():  
    ...  
    return logistic, rf, xgb, preprocessor

@st.cache\_resource  
def get\_tree\_explainer(model):  
    import shap  
    return shap.TreeExplainer(model)

* **Tips:**

  * Precompute encoders/transformers and store in a pipeline.

  * Avoid re-fitting models on every rerun.

  * If CPU-bound, consider limiting threads via env vars (e.g., `OMP_NUM_THREADS=1`) on Streamlit Cloud to reduce contention.

---

## **Suggested Repository Structure**

project2-churn-prediction/  
├── README.md  
├── AI\_USAGE.md  
├── requirements.txt  
├── data/  
│   ├── raw/  
│   └── processed/  
├── src/  
│   ├── data\_prep.py  
│   ├── clv\_analysis.py  
│   ├── train\_models.py  
│   ├── interpretability.py  
│   └── predict.py  
├── models/  
│   ├── logistic.pkl  
│   ├── rf.pkl  
│   └── xgb.pkl  
├── app.py  
└── notebooks/  
    └── exploration.ipynb (optional)

---

## **Video — 2-3 minutes**

Record a concise demo that a hiring manager would understand.

**Suggested flow:**

* **0:00–0:30** – Problem & value: “We predict churn and estimate value (CLV) to prioritize retention.”

* **0:30–1:00** – Live prediction: enter inputs → see probability, risk label, CLV.

* **1:00–2:00** – Explainability & results:

  * Show SHAP (or fallback importance) briefly

  * Show model comparison table & ROC

  * Show CLV quartiles vs churn and 1 key insight

* **2:00-2:30** – Explain your use of AI; include the **public app URL** & **repo**.

Upload to YouTube or Loom. Send me the link.

---

## **Grading Rubric**

**Technical (60%)**

* Clean data prep \+ stratified split (10%)

* CLV computed, quartiles & churn by segment (10%)

* Three models trained, tuned lightly, and evaluated with Precision/Recall/F1/AUC (20%)

* Interpretability works (SHAP **or** documented fallback) (10%)

* Streamlit app deployed, fast, and stable (10%)

**Communication (25%)**

* README: business framing \+ CLV assumptions \+ how to run (10%)

* Clear model comparison & threshold reasoning (5%)

* App copy is concise and non-jargony (5%)

* AI\_USAGE shows critical use (what you verified/fixed) (5%)

**Business Value (15%)**

* Insightful CLV × churn findings (10%)

* Actionable recommendation (who to retain first & why) (5%)

---

## **Data Source**

IBM Telco Customer Churn  
 Direct CSV (no account):  
 `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

---

## **Submission Checklist**

* Processed splits saved

* CLV analysis \+ segments \+ insights

* Logistic, RF, XGBoost trained & evaluated

* SHAP or feature-importance fallback implemented

* Streamlit app deployed \+ public URL

* `requirements.txt`, `README.md`, `AI_USAGE.md`

* 2-3 minute video (link in README)

* Public GitHub repository

---

# **Dependency Management Notes**

**requirements.txt (recommended versions)**

pandas==2.2.\*  
numpy==1.26.\*  
scikit-learn==1.4.\*  
xgboost==2.0.\*  
shap==0.44.\*  
matplotlib==3.8.\*  
streamlit==1.36.\*  
joblib==1.4.\*

**Notes**

* If SHAP causes deployment issues, remove `shap` and rely on the **feature-importance fallback** (document this in AI\_USAGE.md).

