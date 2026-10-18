# DengueWatch LK

**Explainable Machine Learning for Next-Week Dengue Case Forecasting in Sri Lanka**

**Student No: 269161X**

## Project

This project predicts the number of dengue cases expected in the next reporting week for Sri Lankan reporting areas.

The project compares:

- Naive last-week baseline
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

The selected machine-learning model is explained using SHAP and LIME.

## Dataset

The project uses weekly Sri Lankan dengue case counts by reporting area.

Main fields in the original dataset:

- `year`
- `week`
- `start.date`
- `end.date`
- `district`
- `cases`

The notebook creates lagged case features, rolling averages, recent trend features and seasonal features.

## Train / Validation / Test

- Training: up to 2022
- Validation: 2023–2024
- Test: 2025

A chronological split is used because this is a forecasting problem.

## Evaluation

Models are compared using:

- MAE
- RMSE
- R²

The model is selected using the validation results.

## Explainable AI

The notebook includes:

- SHAP global explanation
- SHAP local explanation
- LIME local explanation

## Streamlit App

## Live Application

The deployed Streamlit application is available here:

https://denguewatch-lk-explainable-ml-sov7nqxg475kxbbgkktgrzy.streamlit.app/

The app allows the user to:

- Select a Sri Lankan reporting area
- View the next-week dengue case forecast
- View recent dengue trends
- View a local SHAP explanation
- View model-comparison results

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `DengueWatch_LK_Explainable_ML_269161X.ipynb` - main notebook
- `srilanka_weekly_data.csv` - original dataset
- `best_model.joblib` - trained model
- `dengue_features.csv` - engineered data used by the app
- `model_results.csv` - model-comparison results
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies

## Note

This is an academic prototype and not an official dengue warning system.
