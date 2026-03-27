from pathlib import Path
import json

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Track C - Epidemic Spread", layout="wide")
st.title("Epidemic Spread Prediction Dashboard")


def resolve_artifact(name: str) -> Path:
    candidates = [ARTIFACTS_DIR / name, PROJECT_ROOT / name, SCRIPT_DIR / name]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@st.cache_resource
def load_models():
    try:
        fc = joblib.load(resolve_artifact("track_c_forecaster.pkl"))
        cl = joblib.load(resolve_artifact("track_c_classifier.pkl"))
        return fc, cl
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_data():
    try:
        from track_c_data_loader import load_combined

        data_path = PROJECT_ROOT / "data" / "time_series_covid19_confirmed_global.csv"
        owid_path = PROJECT_ROOT / "data" / "owid-covid-data.csv"
        if data_path.exists():
            return load_combined(str(data_path), str(owid_path))
        return load_combined()
    except Exception as e:
        st.error(f"Data load error: {e}")
        return None


@st.cache_data
def load_results():
    try:
        with open(resolve_artifact("track_c_results.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


fc_model, cl_model = load_models()
df = load_data()
results = load_results()

tab1, tab2, tab3 = st.tabs(["Case Forecast", "Risk Map", "Model Performance"])

with tab1:
    st.subheader("Country-level Case Forecast")
    if df is not None:
        countries = sorted(df["Country/Region"].unique())
        country = st.selectbox(
            "Select country",
            countries,
            index=countries.index("India") if "India" in countries else 0,
        )
        cdf = df[df["Country/Region"] == country].sort_values("date")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(cdf["date"], cdf["confirmed"], color="#378ADD", linewidth=1.5)
        axes[0].set_title(f"Cumulative confirmed - {country}")
        axes[0].set_ylabel("Total cases")

        axes[1].bar(cdf["date"], cdf["daily_new"], color="#D85A30", alpha=0.6, width=1, label="Daily new")
        axes[1].plot(cdf["date"], cdf["rolling_7d"], color="#533AB7", linewidth=2, label="7-day avg")
        axes[1].set_title("Daily new cases")
        axes[1].set_ylabel("Cases")
        axes[1].legend()

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Risk level over time")
        risk_counts = cdf["risk_label"].value_counts()
        c1, c2, c3 = st.columns(3)
        for col, label in zip([c1, c2, c3], ["Low", "Medium", "High"]):
            count = risk_counts.get(label, 0)
            pct = round(100 * count / len(cdf), 1)
            col.metric(label, f"{pct}% of days")

with tab2:
    st.subheader("Global Risk Snapshot - latest available date")
    if df is not None:
        latest = df.groupby("Country/Region").last().reset_index()
        risk_df = latest[["Country/Region", "rolling_7d", "risk_label", "confirmed"]].copy()
        risk_df = risk_df.sort_values("rolling_7d", ascending=False).head(40)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        colors = [
            {"Low": "#639922", "Medium": "#BA7517", "High": "#A32D2D"}.get(str(r), "#888780")
            for r in risk_df["risk_label"]
        ]
        ax2.barh(risk_df["Country/Region"], risk_df["rolling_7d"], color=colors)
        ax2.set_xlabel("7-day avg daily new cases")
        ax2.set_title("Top 40 countries by recent case load")
        ax2.invert_yaxis()

        from matplotlib.patches import Patch

        legend = [
            Patch(color="#A32D2D", label="High"),
            Patch(color="#BA7517", label="Medium"),
            Patch(color="#639922", label="Low"),
        ]
        ax2.legend(handles=legend)
        plt.tight_layout()
        st.pyplot(fig2)

with tab3:
    if results:
        st.subheader("Forecaster performance")
        f = results.get("forecaster", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{f.get('mae', 'N/A')} cases")
        c2.metric("RMSE", f"{f.get('rmse', 'N/A')} cases")
        c3.metric("R2", f.get("r2", "N/A"))

        st.subheader("Risk classifier performance")
        c = results.get("classifier", {})
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", c.get("accuracy", "N/A"))
        c2.metric("Macro F1", c.get("macro_f1", "N/A"))
    else:
        st.info("Run track_c_model.py first to see results.")
