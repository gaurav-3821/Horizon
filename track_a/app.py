# RUN COMMAND: streamlit run track_a/app.py
import sys
import os
import base64
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

TRACK_A_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(TRACK_A_DIR, "artifacts")
MODELS_PATH = os.path.join(ARTIFACTS_DIR, "tabular_models.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "tabular_features.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "tabular_metrics.json")
SHAP_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
TRAINING_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "training_results.png")

BACKGROUND = "#f5f5f0"
CARD_BG = "#ffffff"
BORDER = "#000000"
ACCENT = "#0066cc"
TEXT = "#0a0a0f"
SAFE = "#006600"
TOXIC = "#cc0000"

SCALAR_FEATURES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "NumSaturatedRings",
    "NumAliphaticRings",
]

INPUT_FEATURES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
]

TARGET_COLS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

st.set_page_config(page_title="Horizon | Track A", layout="wide", page_icon="A")

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    *, *::before, *::after {{
        border-radius: 0px !important;
    }}
    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, sans-serif;
    }}
    .stApp {{
        background: {BACKGROUND};
        color: {TEXT};
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    [data-testid="stHeader"] {{display:none;}}
    [data-testid="stSidebar"] {{
        background: #eeeeea;
        border-right: 2px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}
    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 1.25rem;
        max-width: 1800px;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: {TEXT};
        margin-bottom: 0.2rem;
    }}
    .hero-subtitle {{
        color: {TEXT};
        font-size: 1.02rem;
        margin-bottom: 0.75rem;
    }}
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, {ACCENT}, transparent);
        margin-bottom: 0.9rem;
    }}
    .warning-banner {{
        background: #fff3cd;
        border: 2px solid {BORDER};
        color: #856404;
        padding: 12px 16px;
        margin-bottom: 1rem;
        font-weight: 600;
        box-shadow: 4px 4px 0px {BORDER};
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 2px solid {BORDER};
        border-radius: 6px !important;
        padding: 18px 20px;
        min-height: 122px;
        transition: all 0.2s ease-in-out;
        box-shadow: none;
    }}
    .metric-card:hover {{
        box-shadow: 8px 8px 0px {BORDER};
        transform: translate(-3px, -3px);
    }}
    .metric-icon {{
        font-size: 1.25rem;
        margin-bottom: 10px;
        color: {TEXT};
    }}
    .metric-label {{
        color: {TEXT};
        font-size: 0.86rem;
        margin-bottom: 8px;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {TEXT};
        font-size: 0.82rem;
        margin-top: 6px;
    }}
    .section-title {{
        font-size: 1.02rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {TEXT};
        margin-top: 1.5rem;
        margin-bottom: 0.9rem;
        text-align: left;
        width: 100%;
        display: block;
    }}
    .section-title.centered {{
        text-align: center !important;
    }}
    .small-note {{
        color: {TEXT};
        font-size: 0.92rem;
        line-height: 1.6;
    }}
    .result-pill {{
        display: inline-block;
        font-weight: 800;
        padding: 4px 10px;
        border: 2px solid {BORDER};
        box-shadow: 3px 3px 0px {BORDER};
    }}
    .result-pill.safe {{
        color: {SAFE};
        background: #e9f7ea;
    }}
    .result-pill.toxic {{
        color: {TOXIC};
        background: #fdecec;
    }}
    .img-card {{
        background: {CARD_BG};
        border: 2px solid {ACCENT};
        border-radius: 6px !important;
        padding: 18px 18px 12px 18px;
        box-shadow: none;
        transition: all 0.2s ease-in-out;
        margin-top: 1rem;
    }}
    .img-card:hover {{
        box-shadow: 8px 8px 0px {ACCENT};
        transform: translate(-3px, -3px);
    }}
    .img-card img {{
        width: 100%;
        height: auto;
        display: block;
        margin-top: 0.35rem;
    }}
    div[data-baseweb="input"] > div,
    .stNumberInput div[data-baseweb="input"] > div {{
        background: #ffffff !important;
        border: 2px solid {ACCENT} !important;
        color: {TEXT} !important;
    }}
    input[type="number"] {{
        color: {TEXT} !important;
        background-color: #ffffff !important;
    }}
    label, .stNumberInput label {{
        color: {TEXT} !important;
        font-weight: 600 !important;
    }}
    table, .dataframe {{
        width: 100%;
        border-collapse: collapse !important;
        background: {CARD_BG} !important;
    }}
    th, td, .dataframe th, .dataframe td {{
        border: 2px solid {BORDER} !important;
        padding: 10px 12px !important;
        text-align: left !important;
        color: {TEXT} !important;
    }}
    th, .dataframe th {{
        background: #eef5ff !important;
        font-weight: 800 !important;
    }}
</style>
"""


@st.cache_resource
def load_assets():
    models = joblib.load(MODELS_PATH) if os.path.exists(MODELS_PATH) else {}
    feature_names = json.loads(Path(FEATURES_PATH).read_text(encoding="utf-8")) if os.path.exists(FEATURES_PATH) else []
    metrics = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8")) if os.path.exists(METRICS_PATH) else {}
    return models, feature_names, metrics


def load_models():
    models, _, _ = load_assets()
    return models


def render_metric_card(icon: str, label: str, value: str, subtext: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_feature_row(user_inputs: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    row = {name: 0.0 for name in feature_names}
    for name in SCALAR_FEATURES:
        row[name] = float(user_inputs.get(name, 0.0))
    return pd.DataFrame([row], columns=feature_names)


def image_html(path: str) -> str:
    if not os.path.exists(path):
        return '<div class="small-note">Image not found.</div>'
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{os.path.basename(path)}"/>'


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("Horizon")
    st.sidebar.caption("Track navigation")
    st.sidebar.radio(
        "Tracks",
        [
            "Track B — Antibiotic Resistance",
            "Track A — Drug Toxicity",
            "Track C — Epidemic Spread",
        ],
        index=1,
        label_visibility="collapsed",
    )

    try:
        models = load_models()
    except Exception as exc:
        models = {}
        st.error(f"Model load error: {exc}")

    _, feature_names, metrics = load_assets()

    st.markdown('<div class="hero-title">Drug Toxicity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Track A — Tox21 Molecular Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-banner">&#9888; Cloud deployment uses the Track A tabular XGBoost path. Full research pipeline remains local-only.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        render_metric_card("&#129516;", "Model", "XGBoost Tabular", "Deployment-safe molecular scoring")
    with m2:
        render_metric_card("&#129514;", "Compounds", "7,831", "Processed Tox21 records")
    with m3:
        render_metric_card("&#128200;", "Endpoints", "12", "Binary toxicity classifiers")

    left_col, right_col = st.columns([1.0, 1.3], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Molecular Inputs</div>', unsafe_allow_html=True)
        user_inputs: dict[str, float] = {}
        defaults = {
            "MolWt": 250.0,
            "LogP": 2.0,
            "TPSA": 50.0,
            "NumHDonors": 1.0,
            "NumHAcceptors": 3.0,
            "NumRotatableBonds": 4.0,
            "NumAromaticRings": 1.0,
            "RingCount": 1.0,
        }
        for feature in INPUT_FEATURES:
            user_inputs[feature] = st.number_input(feature, value=float(defaults[feature]), step=0.1)

        predict_clicked = st.button("Predict Toxicity Endpoints", use_container_width=True)
        st.markdown(
            '<div class="small-note" style="margin-top:12px; margin-bottom:18px;">Inputs are tabular physicochemical descriptors only. Fingerprint-derived columns remain zero-filled in the cloud interface.</div>',
            unsafe_allow_html=True,
        )
        if metrics:
            avg_auc = np.mean([v["mean_auc"] for v in metrics.values() if v.get("mean_auc") is not None])
            st.markdown(
                f'<div class="small-note" style="margin-top:10px;"><strong>Average CV AUC:</strong> {avg_auc:.4f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="small-note" style="margin-top:10px;"><strong>Status:</strong> Tabular training artifacts are not yet available in this deployment.</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown('<div class="section-title centered">Endpoint Predictions</div>', unsafe_allow_html=True)
        if not models or not feature_names:
            st.info("Tabular model artifacts are not present in this cloud deployment yet.")
        elif predict_clicked:
            try:
                row = build_feature_row(user_inputs, feature_names)
                results_rows = []
                for endpoint in TARGET_COLS:
                    model = models.get(endpoint)
                    if model is None:
                        continue
                    prob = float(model.predict_proba(row)[:, 1][0])
                    label = "Toxic" if prob >= 0.5 else "Safe"
                    label_class = "toxic" if label == "Toxic" else "safe"
                    results_rows.append(
                        {
                            "Endpoint": endpoint,
                            "Risk": f'<span class="result-pill {label_class}">{label.upper()}</span>',
                            "Probability": f"{prob:.3f}",
                        }
                    )
                result_df = pd.DataFrame(results_rows)
                st.markdown(result_df.to_html(index=False, escape=False), unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
        else:
            st.markdown(
                '<div class="small-note">Enter descriptor values and click the prediction button to score all 12 toxicity endpoints.</div>',
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                f'''
                <div class="img-card">
                    <div class="section-title">SHAP Summary</div>
                    {image_html(SHAP_SUMMARY_PATH)}
                </div>
                ''',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'''
                <div class="img-card">
                    <div class="section-title">Training Results</div>
                    {image_html(TRAINING_RESULTS_PATH)}
                </div>
                ''',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
