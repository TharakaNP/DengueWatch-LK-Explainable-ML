import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="DengueWatch LK",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

st.markdown(
    """
    <style>
        :root {
            --ink: #17212b;
            --muted: #667684;
            --line: #e4e9ed;
            --panel: #ffffff;
            --bg: #f5f7f9;
            --accent: #0f766e;
            --accent-soft: #eaf6f4;
        }

        html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .stApp { background: var(--bg); color: var(--ink); }

        /* Main canvas */
        .block-container {
            max-width: 1220px;
            margin: 0 auto;
            padding: 2.25rem 2.25rem 2rem 2.25rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] .block-container {
            padding: 2rem 1.35rem 1.5rem 1.35rem;
        }
        [data-testid="stSidebar"] h3 {
            font-size: 1.12rem;
            margin: 0 0 .15rem 0;
        }
        [data-testid="stSidebar"] hr { margin: 1.2rem 0; }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: .25rem; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            padding: .44rem .45rem;
            border-radius: 8px;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #f3f7f7; }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: var(--accent-soft);
            color: var(--accent);
        }

        /* Typography */
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.025em; }
        h1 { font-size: 2.6rem !important; line-height: 1.1 !important; margin: 0 !important; }
        h2 { font-size: 1.65rem !important; margin-top: 0 !important; }
        h3 { font-size: 1.2rem !important; }

        .dashboard-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(100px, 18%);
            align-items: center;
            gap: 1.5rem;
            padding-bottom: 1.4rem;
            margin-bottom: .4rem;
            border-bottom: 1px solid var(--line);
        }
        .header-visual { width: 100%; max-height: 145px; }
        .dashboard-header h1 { padding: .35rem 0 0; }
        .kicker {
            color: var(--accent);
            font-size: .76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .11em;
        }
        .student-badge {
            color: #5f6d78;
            margin-top: .65rem;
            font-size: .78rem;
        }
        .subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            margin: .55rem 0 0;
            max-width: 760px;
        }
        .section-head { margin-bottom: .25rem; }
        .section-head h2 { margin-bottom: .35rem !important; }
        .section-copy { color: var(--muted); font-size: .95rem; }

        .model-pill {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid #cae8e3;
            border-radius: 999px;
            padding: .32rem .68rem;
            font-size: .78rem;
            font-weight: 700;
            vertical-align: middle;
        }

        /* Inputs */
        [data-baseweb="select"] > div {
            background: #fff !important;
            border-color: #d8e0e5 !important;
            border-radius: 9px !important;
            min-height: 44px !important;
        }
        [data-baseweb="select"] span { color: var(--ink) !important; }

        /* Native metric cards */
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.05rem;
            min-height: 148px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, .025);
        }
        [data-testid="stMetricLabel"] { min-height: 2.6rem; align-items: flex-start; }
        [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 650; line-height: 1.3; }
        [data-testid="stMetricValue"] { color: var(--ink) !important; font-size: clamp(1.55rem, 2.4vw, 2rem); font-variant-numeric: tabular-nums; }

        /* Custom cards */
        .info-card {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.05rem;
        }
        .info-label { color: #71808c; font-size: .82rem; margin-bottom: .2rem; }
        .info-value { color: var(--ink); font-weight: 700; }
        .soft-note {
            background: #eef8f6;
            border: 1px solid #d5ece8;
            border-radius: 10px;
            padding: .95rem 1rem;
            color: #334155;
            line-height: 1.55;
            font-size: .92rem;
        }
        .soft-note strong { color: var(--ink); }
        [data-testid="stMainBlockContainer"] { --st-container-gap: 1rem; }
        [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: .6rem;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }

        /* Misc */
        [data-testid="stAlert"] { border-radius: 10px; }
        .footer-note {
            color: #8795a0;
            font-size: .8rem;
            line-height: 1.5;
            padding-top: .2rem;
        }
        .stButton > button {
            border-radius: 9px;
            font-weight: 650;
        }

        @media (max-width: 900px) {
            .block-container { padding-left: 1.25rem; padding-right: 1.25rem; }
            h1 { font-size: 2.15rem !important; }
        }
        @media (max-width: 640px) {
            .block-container { padding: 1.5rem 1rem; }
            .dashboard-header { gap: .75rem; grid-template-columns: minmax(0, 1fr) 18%; }
            h1 { font-size: 1.85rem !important; }
            .subtitle { font-size: .9rem; }
            .header-visual { min-width: 0; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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

districts = sorted(data["district"].dropna().unique())

with st.sidebar:
    st.markdown("### DengueWatch LK")
    st.caption("Explainable dengue forecasting")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Forecast", "Weekly Trend", "Explainability", "Model Evaluation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model**")
    st.markdown('<span class="model-pill">Extra Trees Regressor</span>', unsafe_allow_html=True)
    st.caption("Next-week regression model")
    st.divider()
    st.caption("MSc in AI · Student No. 269161X")

# Consistent header
st.markdown(
    """
    <div class="dashboard-header">
      <div>
        <div class="kicker">Sri Lanka dengue forecasting</div>
        <h1>DengueWatch LK</h1>
        <div class="subtitle">Next-week dengue case forecasting using Extra Trees with model explanations for transparent interpretation.</div>
        <div class="student-badge">MSc in AI · Student No. 269161X</div>
      </div>
      <svg class="header-visual" viewBox="0 0 190 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Stylized Sri Lanka silhouette with a health analytics chart">
        <rect x="8" y="12" width="174" height="126" rx="22" fill="#eaf6f4"/>
        <path d="M65 24 C62 31 65 37 61 43 L53 53 C52 62 45 67 44 77 C40 89 43 106 51 118 C58 129 69 129 78 122 C89 115 91 102 87 91 C83 77 75 66 72 53 C69 42 71 33 65 24Z" fill="#0f766e"/>
        <rect x="91" y="51" width="78" height="64" rx="9" fill="white" stroke="#b9dcd6"/>
        <path d="M102 64 V102 H158" fill="none" stroke="#17212b" stroke-opacity=".45" stroke-width="2"/>
        <path d="M106 91 L118 84 L130 88 L143 70 L155 75" fill="none" stroke="#0f766e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="143" cy="70" r="3" fill="#17212b"/>
        <path d="M146 25 V39 M139 32 H153" stroke="#0f766e" stroke-width="3" stroke-linecap="round"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Forecast":
    st.markdown('<div class="section-head"><h2>Forecast</h2><div class="section-copy">Select a reporting area to view the latest next-week prediction and recent dengue activity.</div></div>', unsafe_allow_html=True)

    control_col, content_col = st.columns([1, 2], gap="medium")

    with control_col:
        district = st.selectbox("Select reporting area for prediction", districts)
        d = data[data["district"] == district].dropna(subset=FEATURES).sort_values("date").copy()
        if d.empty:
            st.warning("No usable records are available for this reporting area.")
            st.stop()
        latest = d.iloc[-1]
        X_latest = pd.DataFrame([latest[FEATURES].to_dict()])
        prediction = max(0.0, float(model.predict(X_latest)[0]))

    with content_col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">Latest available record</div>
                <div class="info-value">Week {int(latest['week'])} · {latest['date'].date()}</div>
                <div class="info-label" style="margin-top:.35rem">{district}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Next-week forecast", f"{prediction:.0f}")
    m2.metric("Current cases", f"{latest['cases']:.0f}")
    m3.metric("4-week average", f"{latest['rolling_mean_4']:.1f}")
    m4.metric("4-week change", f"{latest['trend_current_4']:+.0f}")

    st.markdown("### Recent weekly cases")
    recent = d.tail(12)[["date", "cases"]].set_index("date")
    st.line_chart(recent, height=300)

    st.markdown(
        '<div class="soft-note"><strong>Academic prototype:</strong> this dashboard is for coursework and demonstration only. It is not an official dengue warning or public-health decision system.</div>',
        unsafe_allow_html=True,
    )

elif page == "Weekly Trend":
    st.markdown('<div class="section-head"><h2>Weekly trend</h2><div class="section-copy">Review the most recent weekly dengue case pattern for a selected reporting area.</div></div>', unsafe_allow_html=True)

    selector_col, _ = st.columns([1, 2], gap="medium")
    with selector_col:
        selected = st.selectbox("Reporting area", districts, key="trend")
    trend = (
        data[data["district"] == selected]
        .sort_values("date")
        .tail(26)[["date", "cases"]]
        .set_index("date")
    )
    st.line_chart(trend, height=460)
    st.caption("Recent weekly observations are shown for the selected reporting area.")
    st.caption("Most recent 26 weekly observations. The forecasting model also uses lagged cases, rolling averages and seasonal week features.")

elif page == "Explainability":
    st.markdown('<div class="section-head"><h2>Explainability</h2><div class="section-copy">Inspect which features pushed the latest local prediction upward or downward.</div></div>', unsafe_allow_html=True)

    selector_col, _ = st.columns([1, 2], gap="medium")
    with selector_col:
        selected = st.selectbox("Reporting area", districts, key="xai")
    dx = data[data["district"] == selected].dropna(subset=FEATURES).sort_values("date")
    latest_xai = dx.iloc[-1]
    X_one = pd.DataFrame([latest_xai[FEATURES].to_dict()])

    summary1, summary2, summary3 = st.columns(3, gap="small")
    summary1.metric("Current cases", f"{latest_xai['cases']:.0f}")
    summary2.metric("4-week average", f"{latest_xai['rolling_mean_4']:.1f}")
    summary3.metric("Latest week", f"{int(latest_xai['week'])}")

    st.markdown("### Local SHAP explanation")
    if st.button("Generate SHAP explanation", type="primary"):
        with st.spinner("Generating SHAP explanation..."):
            preprocessor = model.named_steps["preprocessor"]
            regressor = model.named_steps["model"]
            X_trans = preprocessor.transform(X_one)
            feature_names = preprocessor.get_feature_names_out()
            clean_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer(X_trans)
            named = shap.Explanation(
                values=shap_values.values,
                base_values=shap_values.base_values,
                data=X_trans,
                feature_names=clean_names,
            )
            plt.figure(figsize=(9, 5))
            shap.plots.waterfall(named[0], max_display=10, show=False)
            _, plot_col, _ = st.columns([1, 10, 1], gap="small")
            with plot_col:
                st.pyplot(plt.gcf(), clear_figure=True, width="stretch")
        st.markdown(
            '<div class="soft-note">SHAP values show how individual features pushed this prediction above or below the model’s baseline prediction. These values describe model attribution and should not be interpreted as causal effects.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Generate the explanation when you want to inspect the selected reporting area's latest prediction.")

else:
    st.markdown('<div class="section-head"><h2>Model evaluation</h2><div class="section-copy">Final 2025 test performance of Extra Trees compared with a simple last-week benchmark.</div></div>', unsafe_allow_html=True)

    extra = results[results["Method"].str.contains("Extra Trees", case=False)].iloc[0]
    bench = results[results["Method"].str.contains("benchmark", case=False)].iloc[0]

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("MAE", f"{extra['MAE']:.2f}", delta=f"{extra['MAE'] - bench['MAE']:+.2f} vs benchmark", delta_color="inverse")
    c2.metric("RMSE", f"{extra['RMSE']:.2f}", delta=f"{extra['RMSE'] - bench['RMSE']:+.2f} vs benchmark", delta_color="inverse")
    c3.metric("R²", f"{extra['R2']:.3f}", delta=f"{extra['R2'] - bench['R2']:+.3f} vs benchmark")

    st.markdown("### Test results")
    display_results = results.copy()
    display_results["MAE"] = display_results["MAE"].map(lambda x: f"{x:.2f}")
    display_results["RMSE"] = display_results["RMSE"].map(lambda x: f"{x:.2f}")
    display_results["R2"] = display_results["R2"].map(lambda x: f"{x:.3f}")
    st.dataframe(display_results, width="stretch", hide_index=True)

    st.markdown(
        """
        <div class="soft-note">
            <strong>Interpretation</strong><br>
            Extra Trees has a lower MAE than the last-week benchmark, while the benchmark has a lower RMSE and a higher R². Neither method leads on every metric. Lower MAE and RMSE indicate smaller errors; higher R² indicates a better fit.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    '<div class="footer-note">Inputs: reporting area, week/seasonality, current cases, lagged cases, rolling means and short-term trend. No personal data is used.</div>',
    unsafe_allow_html=True,
)
