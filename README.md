# DengueWatch LK

**Explainable Machine Learning for Next-Week Dengue Case Forecasting in Sri Lanka**

DengueWatch LK is an academic machine learning project developed to forecast the number of dengue cases expected in the following week for Sri Lankan reporting areas.

The project uses historical weekly dengue case data, an **Extra Trees Regressor**, and explainable AI techniques to make the model predictions easier to understand.

---

**Project Objective**

The objective of this project is to predict next-week dengue cases using recent dengue case history, seasonal information, lagged case values, rolling averages, and short-term trends.

This is an academic prototype and is not an official dengue warning or public-health decision system.

---

**Dataset**

The project uses Sri Lankan weekly dengue case data.

- Original records: **26,311**
- Reporting areas: **26**
- Date range: **2006 to 2026**
- ML-ready records after preprocessing and feature engineering: **24,932**
- Missing values in the main columns: **0**
- Duplicate district-date rows: **0**

Main fields include:

- Year
- Week
- Reporting area / district
- Weekly dengue case count

No personal or sensitive data is used.

---

**Feature Engineering**

The model uses the following main features:

- Current weekly dengue case count
- Previous week case counts
- Lagged case values
- 4-week rolling average
- 8-week rolling average
- 12-week rolling average
- 52-week lag
- Short-term case trend
- Week number
- Seasonal sine and cosine features
- Reporting area

The prediction target is the dengue case count for the following week.

---

**Machine Learning Algorithm**

The final machine learning model is **Extra Trees Regressor (Extremely Randomized Trees)**.

Extra Trees was selected because it can model nonlinear relationships between recent dengue case history, seasonal patterns, and reporting areas without assuming a fixed linear relationship.

Compared with a standard Decision Tree, Extra Trees combines many randomized trees. Compared with Random Forest, Extra Trees introduces additional randomness when selecting split thresholds.

---

**Data Split**

A chronological split was used to avoid using future information during model training.

- Training data: up to **2022**
- Validation data: **2023–2024**
- Final test data: **2025**

The 2026 data is partial and is not used for final model evaluation.

---

**Final Model Performance**

**Extra Trees Regressor**

| Metric | Result |
|---|---:|
| MAE | 8.3734 |
| RMSE | 16.1495 |
| R² | 0.9272 |

**Last-week Benchmark**

| Metric | Result |
|---|---:|
| MAE | 8.7744 |
| RMSE | 15.7911 |
| R² | 0.9304 |

Extra Trees achieved a lower MAE than the last-week benchmark. The benchmark remained slightly better on RMSE and R². This shows that dengue case counts have strong week-to-week persistence.

---

**Explainable AI**

The project uses **SHAP** and **LIME** for model interpretation.

**SHAP**

SHAP is used to identify which features have the strongest influence on model predictions and to explain individual predictions.

The most influential features include:

- Current cases
- 4-week rolling average
- Previous week cases
- Recent lagged case values
- Short-term trend

**LIME**

LIME is used to provide a local explanation for an individual prediction by showing how nearby feature values contribute to that prediction.

SHAP and LIME describe model behavior and should not be interpreted as causal effects.

---

**Streamlit Application**

**Live app:** [Open DengueWatch LK](https://denguewatch-lk-explainable-ml-sov7nqxq475kxbgkktqrzy.streamlit.app/)

The project includes a Streamlit dashboard with four sections:

- **Forecast** – shows the next-week prediction for a selected reporting area
- **Weekly Trend** – displays recent weekly dengue case patterns
- **Explainability** – provides local SHAP explanations
- **Model Evaluation** – shows final test results and benchmark comparison

---

**Run the Project Locally**

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

**Main Project Files**

- `app.py` – Streamlit application
- `DengueWatch_LK_ExtraTrees_269161X.ipynb` – final machine learning notebook
- `best_model.joblib` – trained Extra Trees model
- `srilanka_weekly_data.csv` – dengue dataset
- `model_results.csv` – final evaluation results
- `requirements.txt` – Python dependencies
- `.streamlit/config.toml` – Streamlit configuration

---

**Technologies Used**

Python, pandas, NumPy, scikit-learn, SHAP, LIME, Matplotlib, and Streamlit.

---

**Student**

K.L.T.N. Perera  
Student No. **269161X**  
MSc in Artificial Intelligence

---

**Disclaimer**

This project was developed for academic purposes. The forecasts should not be used as official dengue alerts, medical advice, or public-health decision support.
