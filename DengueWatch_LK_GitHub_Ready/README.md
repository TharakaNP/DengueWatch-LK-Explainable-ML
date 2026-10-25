# DengueWatch LK — Extra Trees Forecasting

MSc in AI assignment project for next-week dengue case forecasting in Sri Lanka.

## Main algorithm
**Extra Trees Regressor (Extremely Randomized Trees)** — selected as a non-deep-learning algorithm not covered in the module lectures.

## Project workflow
- Public Sri Lankan weekly dengue data
- Lag, rolling-average, trend and seasonal features
- Chronological train/validation/test split
- Extra Trees hyperparameter selection on validation data
- Final evaluation on 2025 test data
- SHAP and LIME explainability
- Streamlit front-end for prediction and local SHAP explanation

## Final 2025 test result
- MAE: 8.37
- RMSE: 16.15
- R²: 0.927

Last-week benchmark: MAE **8.77**, RMSE **15.79**, R² **0.930**.

## Live application

[Open DengueWatch LK](https://denguewatch-lk-explainable-ml-sov7nqxq475kxbgkktqrzy.streamlit.app/)

## Run locally

From `DengueWatch_LK_GitHub_Ready`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, select `DengueWatch_LK_GitHub_Ready/app.py` as the main file.

The application is an academic prototype and is not an official public-health warning system.
