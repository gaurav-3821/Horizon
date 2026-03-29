from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib import error, request

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "stacked_model_tuned.pkl"
SHAP_FEATURES_PATH = ARTIFACTS_DIR / "shap_features.json"
SHAP_GLOBAL_PATH = ARTIFACTS_DIR / "shap_global.png"
APP_METADATA_PATH = ARTIFACTS_DIR / "app_metadata.json"
CLAUDE_MODEL = "claude-sonnet-4-20250514"


st.set_page_config(
    page_title="AI Clinical Advisor",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #0a0a0f;
        color: #f3f6fb;
        font-family: "Inter", system-ui, sans-serif;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.4rem;
        max-width: 1240px;
    }
    [data-testid="stHeader"] {
        background: rgba(10, 10, 15, 0.88);
    }
    [data-testid="stAppViewContainer"] {
        background: #0a0a0f;
    }
    [data-testid="stToolbar"] {
        right: 1rem;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: "Inter", system-ui, sans-serif;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.18rem;
        background: linear-gradient(90deg, #ffffff 0%, #9ee9ff 45%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: #a0a0b0;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.65rem;
    }
    .hero-divider {
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.95) 0%, rgba(0, 212, 255, 0.05) 100%);
        margin-bottom: 1.1rem;
    }
    .warning-banner {
        background: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.24);
        color: #ffd86b;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        color: #a0a0b0;
        font-size: 0.88rem;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .panel {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.24);
    }
    .section-title {
        color: #ffffff;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .risk-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.15rem 1.2rem;
        min-height: 230px;
    }
    .result-card {
        text-align: center;
        padding-top: 1rem;
    }
    .result-label {
        font-size: 0.82rem;
        letter-spacing: 0.14rem;
        color: #a0a0b0;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }
    .result-probability {
        color: #d6deea;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    .glow-resistant {
        box-shadow: 0 0 0 1px rgba(255, 71, 87, 0.22), 0 0 32px rgba(255, 71, 87, 0.12);
    }
    .glow-susceptible {
        box-shadow: 0 0 0 1px rgba(46, 213, 115, 0.2), 0 0 32px rgba(46, 213, 115, 0.1);
    }
    .meter-wrap {
        background: #1a1a2e;
        border-radius: 999px;
        overflow: hidden;
        height: 18px;
        margin-top: 0.45rem;
        margin-bottom: 0.6rem;
        border: 1px solid #1e1e2e;
    }
    .meter-bar {
        height: 100%;
        border-radius: 999px;
    }
    .small-note {
        color: #a0a0b0;
        font-size: 0.9rem;
    }
    .shap-bar-list {
        display: grid;
        gap: 0.72rem;
        margin-top: 0.55rem;
    }
    .shap-row {
        display: grid;
        gap: 0.35rem;
    }
    .shap-topline {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-size: 0.92rem;
    }
    .shap-name {
        color: #eaf3fb;
        font-weight: 600;
    }
    .shap-value {
        color: #9ddfff;
        font-variant-numeric: tabular-nums;
    }
    .shap-track {
        background: #1a1a2e;
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid #1e1e2e;
    }
    .shap-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00d4ff 0%, #6ce7ff 100%);
    }
    .advisor-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-left: 4px solid #00d4ff;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 0.8rem;
        white-space: pre-wrap;
        color: #eaf3fb;
    }
    .stButton > button {
        background: #00d4ff;
        color: #071018;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.75rem 1rem;
        transition: all 0.18s ease-in-out;
        box-shadow: 0 10px 24px rgba(0, 212, 255, 0.16);
    }
    .stButton > button:hover {
        background: #6ce7ff;
        color: #04080d;
        transform: translateY(-1px);
    }
    .stButton > button:focus {
        outline: 2px solid rgba(0, 212, 255, 0.35);
        outline-offset: 1px;
    }
    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    label, [data-testid="stWidgetLabel"] {
        color: #a0a0b0 !important;
        font-weight: 500 !important;
    }
    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div,
    .stTextInput div[data-baseweb="input"] > div {
        background: #1a1a2e !important;
        border: 1px solid #1e1e2e !important;
        border-radius: 10px !important;
        color: #f3f6fb !important;
    }
    .stSelectbox [data-baseweb="select"] input,
    .stNumberInput input,
    .stTextInput input {
        color: #f3f6fb !important;
    }
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stNumberInput div[data-baseweb="input"] > div:focus-within,
    .stTextInput div[data-baseweb="input"] > div:focus-within {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.2) !important;
    }
    [data-testid="stImage"] img,
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #1e1e2e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_fill_categorical(series: pd.Series, fallback: str) -> pd.Series:
    return (
        series.astype(object)
        .where(series.notna(), fallback)
        .astype(str)
        .replace({"nan": fallback, "": fallback})
    )


def get_feature_names(preprocessor: dict) -> list[str]:
    return preprocessor["categorical_columns"] + preprocessor["numeric_columns"]


def transform_features(X: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    feature_columns = preprocessor["feature_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    numeric_columns = preprocessor["numeric_columns"]

    X = X.copy().reindex(columns=feature_columns)
    parts = []

    if categorical_columns:
        cat_values = preprocessor["cat_imputer"].transform(X[categorical_columns])
        cat_encoded = preprocessor["cat_encoder"].transform(cat_values)
        parts.append(cat_encoded)

    if numeric_columns:
        num_values = preprocessor["num_imputer"].transform(X[numeric_columns])
        parts.append(num_values)

    if not parts:
        raise ValueError("No feature columns available for inference.")

    return np.hstack(parts).astype(np.float32)


@st.cache_resource
def load_runtime_resources():
    missing = [
        path.name
        for path in [MODEL_PATH, SHAP_FEATURES_PATH, SHAP_GLOBAL_PATH, APP_METADATA_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Track B artifacts: {', '.join(missing)}")

    model_bundle = joblib.load(MODEL_PATH)
    with open(SHAP_FEATURES_PATH, "r", encoding="utf-8") as handle:
        shap_features = json.load(handle)
    shap_global_bytes = SHAP_GLOBAL_PATH.read_bytes()

    with open(APP_METADATA_PATH, "r", encoding="utf-8") as handle:
        app_metadata = json.load(handle)

    xgb_model = model_bundle["model"].named_estimators_["xgb"]
    explainer = shap.TreeExplainer(xgb_model)

    return {
        "model_bundle": model_bundle,
        "shap_features": shap_features,
        "shap_global_bytes": shap_global_bytes,
        "species_options": app_metadata["species_options"],
        "site_options": app_metadata["site_options"],
        "sample_type_options": app_metadata["sample_type_options"],
        "antibiotic_options": app_metadata["antibiotic_options"],
        "antibiotic_class_options": app_metadata["antibiotic_class_options"],
        "antibiotic_class_mode": app_metadata["antibiotic_class_mode"],
        "aro_match_median": app_metadata["aro_match_median"],
        "explainer": explainer,
    }


def build_inference_frame(form_values: dict, resources: dict) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    bundle = resources["model_bundle"]
    df_input = pd.DataFrame([form_values])

    df_input["source_dataset"] = "mendeley"
    df_input["species_clean"] = safe_fill_categorical(df_input.get("species"), "unknown_species")
    df_input["antibiotic_class_clean"] = safe_fill_categorical(df_input.get("antibiotic_class"), "unknown_class")
    df_input["antibiotic_species_interaction"] = (
        df_input["antibiotic_class_clean"] + "__" + df_input["species_clean"]
    )

    feature_engineering = bundle["feature_engineering"]
    global_rate = float(feature_engineering["global_rate"])
    class_rate = feature_engineering["antibiotic_class_rate_map"]
    species_rate = feature_engineering["species_rate_map"]

    df_input["antibiotic_class_resistance_rate"] = (
        df_input["antibiotic_class_clean"].map(class_rate).fillna(global_rate)
    )
    df_input["species_resistance_rate"] = df_input["species_clean"].map(species_rate).fillna(global_rate)
    df_input["aro_antibiotic_class"] = df_input["antibiotic_class"]
    df_input["aro_match_count"] = float(resources["aro_match_median"].get(form_values["antibiotic_class"], 0.0))
    df_input["fasta_sequence_found"] = 0.0

    for column in bundle["selected_kmer_columns"]:
        df_input[column] = 0.0

    feature_frame = df_input[bundle["feature_columns"]].copy()
    transformed = transform_features(feature_frame, bundle["preprocessor"])
    feature_names = get_feature_names(bundle["preprocessor"])
    return df_input, transformed, feature_names


def call_claude_clinical_advisor(api_key: str, payload: dict) -> str:
    prompt = f"""
You are an antimicrobial stewardship clinical advisor helping interpret a research model output.

Patient and sample inputs:
- Species: {payload['species']}
- Antibiotic: {payload['antibiotic_name']}
- Antibiotic class: {payload['antibiotic_class']}
- Age: {payload['age']}
- Gender: {payload['gender']}
- Site: {payload['site']}
- Sample type: {payload['sample_type']}
- Prior hospitalisation: {payload['Hospital_before']}
- Hypertension: {payload['Hypertension']}
- Diabetes: {payload['Diabetes']}
- Infection frequency: {payload['Infection_Freq']}

Model output:
- Prediction: {payload['prediction_label']}
- Confidence: {payload['confidence_pct']}%
- Resistant probability: {payload['resistant_probability_pct']}%

Top model features for this prediction:
{json.dumps(payload['top_shap_features'], indent=2)}

Provide:
1. A short clinical interpretation of the model output.
2. Key patient or microbiology factors driving concern.
3. Practical antibiotic stewardship recommendations or cautionary next steps.
4. A short note on uncertainty and why culture/lab validation is still required.

Keep it concise, clinician-facing, and clearly structured.
"""

    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 700,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Anthropic API error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Anthropic API connection failed: {exc.reason}") from exc

    content = raw.get("content", [])
    text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n\n".join(part for part in text_parts if part).strip()


def render_meter(confidence: float, label: str):
    pct = max(0.0, min(100.0, confidence * 100.0))
    color = "#ff4757" if label == "Resistant" else "#2ed573"
    st.markdown(
        f"""
        <div class="meter-wrap">
            <div class="meter-bar" style="width: {pct:.1f}%; background: {color};"></div>
        </div>
        <div class="small-note">Confidence meter: {pct:.1f}%</div>
        """,
        unsafe_allow_html=True,
    )


def render_top_shap_bars(features: list[dict]):
    if not features:
        st.info("No local SHAP features available for this prediction.")
        return

    max_abs = max(feature["abs_shap"] for feature in features) or 1.0
    rows = []
    for feature in features:
        width = max(8.0, (feature["abs_shap"] / max_abs) * 100.0)
        rows.append(
            f"""
            <div class="shap-row">
                <div class="shap-topline">
                    <span class="shap-name">{feature['feature']}</span>
                    <span class="shap-value">{feature['shap_value']:.4f}</span>
                </div>
                <div class="shap-track">
                    <div class="shap-fill" style="width: {width:.1f}%"></div>
                </div>
            </div>
            """
        )
    st.markdown(f'<div class="shap-bar-list">{"".join(rows)}</div>', unsafe_allow_html=True)


try:
    resources = load_runtime_resources()
except Exception as exc:
    st.error(f"Failed to load Track B runtime artifacts: {exc}")
    st.stop()


st.markdown('<div class="hero-title">AI Clinical Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Track B — Antibiotic Resistance Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="warning-banner">
    ⚠️ Model trained on Mendeley AMR dataset. For research use only. Validate before clinical deployment.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Model AUC</div>
            <div class="metric-value">0.8540</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">1,370</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient &amp; Sample Inputs</div>', unsafe_allow_html=True)

    default_antibiotic = resources["antibiotic_options"][0] if resources["antibiotic_options"] else "IMIPENEM"
    default_class = resources["antibiotic_class_mode"].get(default_antibiotic, "carbapenem")

    with st.form("clinical_input_form", border=False):
        species = st.selectbox(
            "Species",
            options=resources["species_options"] or ["unknown"],
            index=0,
        )
        antibiotic_name = st.selectbox(
            "Antibiotic Name",
            options=resources["antibiotic_options"] or ["IMIPENEM"],
            index=(
                resources["antibiotic_options"].index(default_antibiotic)
                if default_antibiotic in resources["antibiotic_options"]
                else 0
            ),
        )
        default_class = resources["antibiotic_class_mode"].get(antibiotic_name, default_class)
        antibiotic_class = st.selectbox(
            "Antibiotic Class",
            options=resources["antibiotic_class_options"] or ["carbapenem"],
            index=(
                resources["antibiotic_class_options"].index(default_class)
                if default_class in resources["antibiotic_class_options"]
                else 0
            ),
        )

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
            gender = st.selectbox("Gender", ["F", "M", "Unknown"])
            site = st.selectbox("Site", options=resources["site_options"] or ["IFE"], index=0)
            Hospital_before = st.selectbox("Hospital Before", ["Yes", "No", "Unknown"])
            Hypertension = st.selectbox("Hypertension", ["Yes", "No", "Unknown"])
        with col2:
            sample_type = st.selectbox("Sample Type", options=resources["sample_type_options"] or ["T"], index=0)
            Diabetes = st.selectbox("Diabetes", ["Yes", "No", "Unknown"])
            Infection_Freq = st.selectbox("Infection Frequency", ["First", "Recurrent", "Unknown"])

        predict = st.form_submit_button("Predict Resistance", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
    st.image(
        BytesIO(resources["shap_global_bytes"]),
        caption="Global SHAP summary (XGBoost base model)",
        use_container_width=True,
    )
    st.caption("Top global drivers from the tuned Mendeley-only model")
    global_df = pd.DataFrame(resources["shap_features"])
    st.dataframe(global_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


if predict:
    form_values = {
        "species": species,
        "antibiotic_name": antibiotic_name,
        "antibiotic_class": antibiotic_class,
        "age": float(age),
        "gender": gender,
        "site": site,
        "sample_type": sample_type,
        "Hospital_before": Hospital_before,
        "Hypertension": Hypertension,
        "Diabetes": Diabetes,
        "Infection_Freq": Infection_Freq,
    }

    try:
        _, transformed_row, feature_names = build_inference_frame(form_values, resources)
        model_bundle = resources["model_bundle"]
        probability_resistant = float(model_bundle["model"].predict_proba(transformed_row)[0, 1])
        prediction_label = "Resistant" if probability_resistant >= 0.5 else "Susceptible"
        confidence = probability_resistant if prediction_label == "Resistant" else (1.0 - probability_resistant)

        shap_values = resources["explainer"].shap_values(transformed_row)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        sample_shap = np.asarray(shap_values)[0]
        top_local_idx = np.argsort(np.abs(sample_shap))[::-1][:5]
        top_local_features = [
            {
                "feature": feature_names[idx],
                "shap_value": float(sample_shap[idx]),
                "abs_shap": float(abs(sample_shap[idx])),
            }
            for idx in top_local_idx
        ]

        st.session_state["track_b_result"] = {
            "form_values": form_values,
            "prediction_label": prediction_label,
            "confidence": confidence,
            "probability_resistant": probability_resistant,
            "top_local_features": top_local_features,
        }
    except Exception as exc:
        st.error(f"Inference failed: {exc}")


result = st.session_state.get("track_b_result")

if result:
    st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.95, 1.05], gap="large")

    with c1:
        result_color = "#ff4757" if result["prediction_label"] == "Resistant" else "#2ed573"
        glow_class = "glow-resistant" if result["prediction_label"] == "Resistant" else "glow-susceptible"
        st.markdown(f'<div class="risk-card result-card {glow_class}">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-value" style="color: {result_color};">{result["prediction_label"].upper()}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-probability">Confidence: {result["confidence"] * 100:.1f}%</div>',
            unsafe_allow_html=True,
        )
        render_meter(result["confidence"], result["prediction_label"])
        st.markdown(
            f"<div class='small-note'>Resistant probability: {result['probability_resistant'] * 100:.1f}%</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="risk-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='small-note'>Top 5 local feature contributions from the XGBoost base model.</div>",
            unsafe_allow_html=True,
        )
        render_top_shap_bars(result["top_local_features"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">AI Clinical Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
        Generate a clinician-facing interpretation using Claude. This uses patient inputs, the model output,
        confidence score, and the top local SHAP features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Generate Clinical Interpretation", use_container_width=True):
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Missing `ANTHROPIC_API_KEY` in Streamlit secrets.")
        else:
            payload = {
                **result["form_values"],
                "prediction_label": result["prediction_label"],
                "confidence_pct": round(result["confidence"] * 100, 1),
                "resistant_probability_pct": round(result["probability_resistant"] * 100, 1),
                "top_shap_features": result["top_local_features"],
            }
            try:
                with st.spinner("Consulting Claude..."):
                    advisor_text = call_claude_clinical_advisor(api_key, payload)
                st.markdown(f'<div class="advisor-card">{advisor_text}</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Clinical advisor request failed: {exc}")
