import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="DengueWatch LK",
    page_icon="🦟",
    layout="wide"
)

BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE / "best_model.joblib")

@st.cache_data
def load_data():
    data = pd.read_csv(BASE / "dengue_features.csv", parse_dates=["date"])
    results = pd.read_csv(BASE / "model_results.csv")
    return data, results

model = load_model()
data, results = load_data()

FEATURES = [
    "district", "week", "week_sin", "week_cos", "cases",
    "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4",
    "cases_lag_8", "cases_lag_12", "cases_lag_52",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_12",
    "trend_current_4"
]

st.title("🦟 DengueWatch LK")
st.write("Explainable Machine Learning for Next-Week Dengue Case Forecasting in Sri Lanka")
st.caption("Student No: 269161X")

st.info(
    "This is an academic prototype. It is not an official dengue warning system "
    "and should not be used for medical or public-health decisions."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Recent Trend",
    "Explainable AI",
    "Model Comparison"
])

districts = sorted(data["district"].dropna().unique())

with tab1:
    district = st.selectbox("Select reporting area", districts)

    d = (
        data[data["district"] == district]
        .dropna(subset=FEATURES)
        .sort_values("date")
        .copy()
    )

    if d.empty:
        st.warning("No usable records are available for this reporting area.")
        st.stop()

    latest = d.iloc[-1]
    X_latest = pd.DataFrame([latest[FEATURES].to_dict()])
    prediction = max(0.0, float(model.predict(X_latest)[0]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted cases next week", f"{prediction:.0f}")
    c2.metric("Current week cases", f"{latest['cases']:.0f}")
    c3.metric("4-week average", f"{latest['rolling_mean_4']:.1f}")
    c4.metric("Change vs 4 weeks ago", f"{latest['trend_current_4']:+.0f}")

    st.write(
        f"Latest available record for **{district}**: "
        f"week {int(latest['week'])}, {latest['date'].date()}."
    )

with tab2:
    selected_district = st.selectbox(
        "Reporting area",
        districts,
        key="trend_district"
    )

    trend_data = (
        data[data["district"] == selected_district]
        .sort_values("date")
        .tail(20)[["date", "cases"]]
        .set_index("date")
    )

    st.line_chart(trend_data)
    st.caption("Most recent 20 weekly observations.")

with tab3:
    selected_district_xai = st.selectbox(
        "Reporting area",
        districts,
        key="xai_district"
    )

    dx = (
        data[data["district"] == selected_district_xai]
        .dropna(subset=FEATURES)
        .sort_values("date")
    )

    latest_xai = dx.iloc[-1]
    X_one = pd.DataFrame([latest_xai[FEATURES].to_dict()])

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    X_trans = preprocessor.transform(X_one)
    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer(X_trans)

    named_explanation = shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=X_trans,
        feature_names=clean_feature_names
    )

    st.subheader("Local SHAP explanation")
    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(named_explanation[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.write(
        "Positive SHAP values push the forecast upward, while negative values "
        "push it downward for this individual prediction."
    )

with tab4:
    st.dataframe(
        results.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R2": "{:.3f}"
        }),
        use_container_width=True
    )

    val = results[results["Split"] == "Validation"].copy()
    ml_val = val[val["Model"] != "Naive last-week"].sort_values("RMSE")

    if not ml_val.empty:
        best = ml_val.iloc[0]
        st.success(
            f"Selected model: {best['Model']} "
            f"(Validation RMSE = {best['RMSE']:.2f}, R² = {best['R2']:.3f})"
        )

st.divider()
st.caption(
    "Features include recent dengue case counts, lagged case counts, rolling averages, "
    "seasonality and reporting area."
)
