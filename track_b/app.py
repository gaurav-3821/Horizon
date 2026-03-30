import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import base64
import html
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.decomposition import PCA

from track_b_model import get_transformed_feature_names, prepare_bundle_features


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "stacked_model_tuned.pkl"
SHAP_FEATURES_PATH = ARTIFACTS_DIR / "shap_features.json"
SHAP_GLOBAL_PATH = ARTIFACTS_DIR / "shap_global.png"
DATA_PATH = ARTIFACTS_DIR / "unified_dataset_final.csv"
KMER_COLUMNS = [
    "kmer4_VRIT",
    "kmer4_ASWV",
    "kmer4_RALV",
    "kmer4_ALVE",
    "kmer4_SVLA",
    "kmer4_ANAS",
    "kmer4_SYVA",
    "kmer4_YTSG",
    "kmer4_GALA",
    "kmer4_FKPL",
    "kmer4_GMAV",
    "kmer4_PGMA",
    "kmer4_FELG",
    "kmer4_ELGS",
    "kmer4_LATY",
    "kmer4_ATYT",
    "kmer4_HKTG",
    "kmer4_LGWE",
    "kmer4_YGVK",
    "kmer4_AYGV",
]

BACKGROUND = "#0a0a0f"
CARD_BG = "#111118"
BORDER = "#1e1e2e"
INPUT_BG = "#1a1a2e"
ACCENT = "#00d4ff"
RESISTANT = "#ff4757"
SUSCEPTIBLE = "#2ed573"
TEXT_MUTED = "#a0a0b0"


st.set_page_config(page_title="Horizon | Track B", layout="wide", page_icon="\U0001F9EC")


CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    *, *::before, *::after {{
        border-radius: 0px !important;
    }}
    html, body, [class*="css"]  {{
        font-family: 'Inter', system-ui, sans-serif;
    }}
    .stApp {{
        background: {BACKGROUND};
        color: #f5f7fb;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    [data-testid="stHeader"] {{display: none;}}
    [data-testid="stSidebar"] {{
        background: #0d0d15;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: #f5f7fb;
    }}
    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 1.25rem;
        max-width: 1800px;
    }}
    .card {{
        background: {CARD_BG};
        border: 2px solid {ACCENT};
        padding: 20px;
        box-shadow: 4px 4px 0px {ACCENT};
        height: 100%;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 2px solid #ffffff;
        padding: 18px 20px;
        min-height: 122px;
        box-shadow: 4px 4px 0px #ffffff;
    }}
    .metric-icon {{
        font-size: 1.25rem;
        margin-bottom: 10px;
        color: {ACCENT};
    }}
    .metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.86rem;
        margin-bottom: 8px;
    }}
    .metric-value {{
        color: #f5f7fb;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {ACCENT};
        font-size: 0.82rem;
        margin-top: 6px;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, {ACCENT} 55%, #7a5cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}
    .hero-subtitle {{
        color: {TEXT_MUTED};
        font-size: 1.02rem;
        margin-bottom: 0.75rem;
    }}
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, {ACCENT}, transparent);
        margin-bottom: 0.9rem;
    }}
    .warning-banner {{
        background: rgba(255, 184, 0, 0.12);
        border: 2px solid #ffa502;
        color: #f5d97b;
        padding: 12px 16px;
        margin-bottom: 1rem;
        font-weight: 600;
        box-shadow: 4px 4px 0px #ffa502;
    }}
    .section-title {{
        font-size: 1.02rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #ffffff;
        margin-bottom: 0.9rem;
    }}
    .section-step {{
        font-size: 0.82rem;
        color: {ACCENT};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }}
    .prediction-text {{
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.3rem 0 0.25rem;
    }}
    .prediction-sub {{
        text-align: center;
        color: {TEXT_MUTED};
        margin-bottom: 0.85rem;
    }}
    .prediction-state {{
        margin-top: 18px;
        padding: 18px;
        border: 3px solid #ffffff;
        box-shadow: 6px 6px 0px #ffffff;
    }}
    .prediction-state.resistant {{
        border: 3px solid {RESISTANT};
        box-shadow: 6px 6px 0px {RESISTANT};
    }}
    .prediction-state.susceptible {{
        border: 3px solid {SUSCEPTIBLE};
        box-shadow: 6px 6px 0px {SUSCEPTIBLE};
    }}
    .progress-shell {{
        width: 100%;
        height: 14px;
        background: #191926;
        overflow: hidden;
        border: 2px solid #ffffff;
    }}
    .progress-fill {{
        height: 100%;
        transition: width 0.3s ease;
    }}
    .local-bar {{
        margin-bottom: 14px;
    }}
    .local-bar-head {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }}
    .local-bar-name {{
        color: #ffffff;
        font-weight: 600;
    }}
    .local-bar-value {{
        color: {TEXT_MUTED};
        font-variant-numeric: tabular-nums;
    }}
    .local-bar-track {{
        width: 100%;
        height: 12px;
        background: #171726;
        border: 2px solid {ACCENT};
        overflow: hidden;
    }}
    .local-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, {ACCENT}, #7a5cff);
        box-shadow: 3px 3px 0px #7a5cff;
    }}
    .advisor-box {{
        background: linear-gradient(135deg, rgba(64, 45, 145, 0.4), rgba(0, 68, 122, 0.35)), {CARD_BG};
        border: 2px solid {ACCENT};
        padding: 20px;
        min-height: 320px;
        box-shadow: 4px 4px 0px {ACCENT};
    }}
    .advisor-card {{
        background: rgba(11, 16, 27, 0.48);
        border: 2px solid {ACCENT};
        padding: 18px;
        height: 100%;
        box-shadow: 4px 4px 0px {ACCENT};
    }}
    .advisor-title {{
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.85rem;
    }}
    .advisor-response-box {{
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid {ACCENT};
        border-left: 4px solid {ACCENT};
        border-radius: 12px;
        padding: 24px;
        margin-top: 14px;
    }}
    .advisor-placeholder {{
        color: {TEXT_MUTED};
        line-height: 1.6;
    }}
    .advisor-response-text {{
        color: #f5f7fb;
        line-height: 1.7;
        white-space: pre-wrap;
    }}
    .pill {{
        display: inline-block;
        border: 2px solid {ACCENT};
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(0, 212, 255, 0.12);
        color: {ACCENT};
        margin-right: 8px;
        margin-bottom: 8px;
        box-shadow: 3px 3px 0px {ACCENT};
    }}
    .small-note {{
        color: {TEXT_MUTED};
        font-size: 0.87rem;
        line-height: 1.55;
    }}
    .stButton > button, .stDownloadButton > button {{
        background: {ACCENT};
        color: #07111a;
        border: 2px solid #000 !important;
        font-weight: 800;
        box-shadow: 3px 3px 0px #000;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: #37defc;
        color: #07111a;
        box-shadow: 1px 1px 0px #000;
        transform: translate(2px, 2px);
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput div[data-baseweb="input"] > div {{
        background: {INPUT_BG} !important;
        border: 2px solid {ACCENT} !important;
        color: #f5f7fb !important;
    }}
    label, .stSelectbox label, .stNumberInput label {{
        color: {TEXT_MUTED} !important;
        font-weight: 600 !important;
    }}
</style>
"""


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not SHAP_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing SHAP features artifact: {SHAP_FEATURES_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing unified dataset: {DATA_PATH}")

    model_bundle = joblib.load(MODEL_PATH)
    with open(SHAP_FEATURES_PATH, "r", encoding="utf-8") as handle:
        shap_features = json.load(handle)
    data = pd.read_csv(DATA_PATH)
    shap_global_b64 = None
    if SHAP_GLOBAL_PATH.exists():
        shap_global_b64 = base64.b64encode(SHAP_GLOBAL_PATH.read_bytes()).decode("utf-8")
    return model_bundle, shap_features, data, shap_global_b64


@st.cache_resource
def build_pca_projection():
    model_bundle, _, data, _ = load_assets()
    mendeley = data[data["source_dataset"].astype(str).str.lower() == "mendeley"].copy()
    if mendeley.empty:
        raise ValueError("No Mendeley rows found for PCA projection.")

    frame = mendeley.copy()
    label_series = frame["resistance_label"].astype(str).str.upper().replace({"I": "S"})
    matrix = prepare_bundle_features(frame, model_bundle)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(matrix)

    projection = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "pc3": coords[:, 2],
            "label_name": np.where(label_series.eq("R"), "Resistant", "Susceptible"),
        }
    )
    return pca, projection


def call_groq_advisor(payload: dict) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Missing `OPENROUTER_API_KEY` in Streamlit secrets. Add the key before using the clinical advisor."

    prompt = f"""
You are a clinical antimicrobial stewardship advisor. Interpret this research model output clearly and concisely.

Case ID: {payload['case_id']}
Patient inputs: {json.dumps(payload['inputs'], indent=2)}
Prediction summary: {json.dumps(payload['model_results'], indent=2)}
Top local SHAP features: {json.dumps(payload['top_local_features'], indent=2)}

Provide:
1. Clinical interpretation
2. Key drivers
3. Stewardship recommendations (2 bullets)
4. Uncertainty note

Keep the output concise, clinician-facing, and structured.
""".strip()

    body = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "provider": {"order": ["Groq"], "allow_fallbacks": False},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/gaurav-3821/Horizon",
            "X-Title": "Horizon AI Clinical Advisor",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return f"OpenRouter API error: {detail}"
    except Exception as exc:
        return f"OpenRouter API call failed: {exc}"


def make_input_frame(inputs: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([inputs])
    for column in KMER_COLUMNS:
        input_df[column] = 0.0
    return input_df


def get_xgb_model_from_stack(stack_model):
    if hasattr(stack_model, "named_estimators_") and "xgb" in stack_model.named_estimators_:
        return stack_model.named_estimators_["xgb"]
    if hasattr(stack_model, "estimators_") and len(stack_model.estimators_) >= 1:
        return stack_model.estimators_[0]
    raise ValueError("Unable to extract XGBoost base model from stacked ensemble.")


def compute_local_shap(model_bundle: dict, input_df: pd.DataFrame):
    matrix = prepare_bundle_features(input_df, model_bundle)
    stack_model = model_bundle["model"]
    xgb_model = get_xgb_model_from_stack(stack_model)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(matrix)
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[1][0])
    else:
        shap_row = np.asarray(shap_values[0])
    feature_names = get_transformed_feature_names(model_bundle["preprocessor"])
    rows = []
    for name, value in zip(feature_names, shap_row):
        rows.append(
            {
                "feature": str(name),
                "value": float(value),
                "abs_value": float(abs(value)),
                "direction": "positive" if value >= 0 else "negative",
            }
        )
    top_rows = sorted(rows, key=lambda item: item["abs_value"], reverse=True)[:5]
    return matrix, top_rows


def render_local_shap_bars(top_features: list[dict]):
    if not top_features:
        st.info("No local SHAP features available.")
        return
    max_value = max(item["abs_value"] for item in top_features) or 1.0
    html_parts = []
    for item in top_features:
        width_pct = max(8.0, (item["abs_value"] / max_value) * 100.0)
        html_parts.append(
            f"""
            <div class="local-bar">
                <div class="local-bar-head">
                    <span class="local-bar-name">{item['feature']}</span>
                    <span class="local-bar-value">{item['value']:+.4f}</span>
                </div>
                <div class="local-bar-track">
                    <div class="local-bar-fill" style="width:{width_pct:.2f}%"></div>
                </div>
            </div>
            """
        )
    html = "".join(html_parts)
    st.markdown(html, unsafe_allow_html=True)


def make_prediction_plot(projection_df: pd.DataFrame, star_coords: np.ndarray | None):
    fig = px.scatter_3d(
        projection_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="label_name",
        color_discrete_map={"Susceptible": SUSCEPTIBLE, "Resistant": RESISTANT},
        template="plotly_dark",
        title="Proximity to Historical Data Clusters",
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=0, r=0, t=46, b=0),
        legend_title_text="Historical label",
        scene=dict(
            bgcolor=CARD_BG,
            xaxis=dict(backgroundcolor=CARD_BG, gridcolor="#2c2c3a", zerolinecolor="#2c2c3a"),
            yaxis=dict(backgroundcolor=CARD_BG, gridcolor="#2c2c3a", zerolinecolor="#2c2c3a"),
            zaxis=dict(backgroundcolor=CARD_BG, gridcolor="#2c2c3a", zerolinecolor="#2c2c3a"),
        ),
    )
    if star_coords is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[star_coords[0]],
                y=[star_coords[1]],
                z=[star_coords[2]],
                mode="markers",
                marker=dict(size=14, color="#ffffff", symbol="diamond", line=dict(color=ACCENT, width=3)),
                name="Current patient",
            )
        )
    return fig


def extract_stewardship_recs(prediction_label: str, resistant_probability: float, top_features: list[dict]) -> list[str]:
    drivers = ", ".join(item["feature"] for item in top_features[:2]) if top_features else "model drivers"
    if prediction_label == "Resistant":
        return [
            f"Escalate stewardship review before empiric use; strongest model drivers: {drivers}.",
            f"Resistant probability is {resistant_probability:.1%}; correlate with culture and susceptibility panel before action.",
        ]
    return [
        f"Prediction favors susceptibility; confirm with culture and local antibiogram before de-escalation.",
        f"Use {drivers} as context only; avoid single-model decisions without microbiology validation.",
    ]


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


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("Horizon")
    st.sidebar.caption("Track navigation")
    st.sidebar.radio(
        "Tracks",
        [
            "\U0001F9EC Track B â€” Antibiotic Resistance",
            "\U0001F9EA Track A â€” Drug Toxicity",
            "\U0001F9A0 Track C â€” Epidemic Spread",
        ],
        index=0,
        label_visibility="collapsed",
    )

    model_bundle, shap_features, unified_df, shap_global_b64 = load_assets()
    pca_model, projection_df = build_pca_projection()
    mendeley_df = unified_df[unified_df["source_dataset"].astype(str).str.lower() == "mendeley"].copy()

    st.markdown('<div class="hero-title">AI Clinical Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Track B â€” Antibiotic Resistance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-banner">&#9888; Model trained on Mendeley AMR dataset. For research use only.</div>',
        unsafe_allow_html=True,
    )

    prediction_payload = st.session_state.get("track_b_prediction_payload")
    latency_display = "-"
    if prediction_payload:
        latency_display = f"{prediction_payload['latency_ms']:.1f} ms"

    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        render_metric_card("&#128202;", "Model AUC", "0.8540", "Validated on Mendeley-only stack")
    with m2:
        render_metric_card("&#129514;", "Training Samples", "1,370", "Mendeley subset only")
    with m3:
        render_metric_card("&#129516;", "Model", "Mendeley Tuned Stack", "XGBoost + LightGBM + CatBoost")
    with m4:
        render_metric_card("&#9889;", "Latency", latency_display, "Prediction path only")

    left_col, center_col, right_col = st.columns([1.1, 2.0, 1.35], gap="large")

    def get_unique_options(frame: pd.DataFrame, column: str, fallback: list[str] | None = None) -> list[str]:
        if column in frame.columns:
            values = (
                frame[column]
                .dropna()
                .astype(str)
                .map(str.strip)
                .replace("", np.nan)
                .dropna()
                .unique()
                .tolist()
            )
            if values:
                return sorted(values)
        return fallback or []

    species_options = get_unique_options(unified_df, "species_clean", ["unknown_species"])
    antibiotic_options = get_unique_options(unified_df, "antibiotic_name")
    class_options = get_unique_options(unified_df, "antibiotic_class")
    gender_options = ["M", "F"]
    site_options = get_unique_options(unified_df, "site")
    sample_type_options = get_unique_options(unified_df, "sample_type")
    yes_no_options = ["Yes", "No"]
    infection_options = ["Low", "Medium", "High"]

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Input Command Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-step">1. Patient Details</div>', unsafe_allow_html=True)
        species = st.selectbox("Species", species_options, index=0 if species_options else None)
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
        gender = st.selectbox("Gender", gender_options, index=0 if gender_options else None)

        st.markdown('<div class="section-step">2. Sample Info</div>', unsafe_allow_html=True)
        antibiotic_name = st.selectbox("Antibiotic", antibiotic_options, index=0 if antibiotic_options else None)
        antibiotic_class = st.selectbox("Antibiotic class", class_options, index=0 if class_options else None)
        site = st.selectbox("Site", site_options, index=0 if site_options else None)
        sample_type = st.selectbox("Sample type", sample_type_options, index=0 if sample_type_options else None)
        hospital_before = st.selectbox("Hospital_before", yes_no_options, index=0 if yes_no_options else None)
        hypertension = st.selectbox("Hypertension", sorted(mendeley_df["Hypertension"].dropna().astype(str).unique().tolist()), index=0)
        diabetes = st.selectbox("Diabetes", sorted(mendeley_df["Diabetes"].dropna().astype(str).unique().tolist()), index=0)
        infection_freq = st.selectbox("Infection_Freq", infection_options, index=0 if infection_options else None)

        st.markdown('<div class="section-step">3. Resistance Prediction</div>', unsafe_allow_html=True)
        predict_clicked = st.button("Predict Resistance", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    current_star = None
    local_top_features = []
    prediction_label = "No prediction"
    resistant_probability = None
    confidence_pct = None
    case_id = None

    if predict_clicked:
        input_payload = {
            "species": species,
            "antibiotic_name": antibiotic_name,
            "antibiotic_class": antibiotic_class,
            "age": int(age),
            "gender": gender,
            "site": site,
            "sample_type": sample_type,
            "Hospital_before": hospital_before,
            "Hypertension": hypertension,
            "Diabetes": diabetes,
            "Infection_Freq": infection_freq,
            "source_dataset": "mendeley",
            "aro_match_count": 0,
            "aro_antibiotic_class": antibiotic_class,
            "fasta_sequence_found": 0,
        }
        input_df = make_input_frame(input_payload)
        start_time = time.time()
        matrix = prepare_bundle_features(input_df, model_bundle)
        resistant_probability = float(model_bundle["model"].predict_proba(matrix)[:, 1][0])
        latency_ms = (time.time() - start_time) * 1000.0
        current_star = pca_model.transform(matrix)[0]
        _, local_top_features = compute_local_shap(model_bundle, input_df)

        prediction_label = "Resistant" if resistant_probability >= 0.5 else "Susceptible"
        confidence_pct = float(max(resistant_probability, 1.0 - resistant_probability) * 100.0)
        case_id = datetime.now().strftime("HZ-%Y-%m%d-%H%M")
        report_dict = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "inputs": input_payload,
            "model_results": {
                "prediction": prediction_label,
                "label": 1 if prediction_label == "Resistant" else 0,
                "confidence_score": round(confidence_pct / 100.0, 4),
                "resistant_probability": round(resistant_probability, 4),
                "model_version": "Mendeley Tuned Stack",
            },
            "top_local_features": local_top_features,
            "latency_ms": round(latency_ms, 2),
            "pca_coords": [float(value) for value in current_star],
        }
        st.session_state["track_b_prediction_payload"] = report_dict
        st.session_state.pop("track_b_advisor_text", None)
        prediction_payload = report_dict
        st.rerun()

    if prediction_payload:
        case_id = prediction_payload["case_id"]
        resistant_probability = prediction_payload["model_results"]["resistant_probability"]
        confidence_pct = prediction_payload["model_results"]["confidence_score"] * 100.0
        prediction_label = prediction_payload["model_results"]["prediction"]
        latency_display = f"{prediction_payload['latency_ms']:.1f} ms"
        local_top_features = prediction_payload["top_local_features"]
        if prediction_payload.get("pca_coords"):
            current_star = np.asarray(prediction_payload["pca_coords"], dtype=float)

    with center_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        figure = make_prediction_plot(projection_df, current_star)
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
        result_color = RESISTANT if prediction_label == "Resistant" else SUSCEPTIBLE
        if prediction_payload:
            state_class = "resistant" if prediction_label == "Resistant" else "susceptible"
            st.markdown(f'<div class="prediction-state {state_class}">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="prediction-text" style="color:{result_color};">{prediction_label.upper()}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="prediction-sub">Confidence {confidence_pct:.1f}% | Resistant probability {resistant_probability:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="progress-shell">
                    <div class="progress-fill" style="width:{confidence_pct:.1f}%; background:{result_color}; box-shadow:0 0 20px {result_color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="prediction-sub">Run a prediction to place the current patient inside the historical cluster map.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Global Drivers</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(shap_features[:10]), use_container_width=True, hide_index=True)
        if shap_global_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{shap_global_b64}" style="width:100%; border-radius:12px; margin-top:12px;" />',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Local Feature Contributions</div>', unsafe_allow_html=True)
        if prediction_payload:
            render_local_shap_bars(local_top_features[:5])
        else:
            st.info("Run a prediction to compute local SHAP contributions.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="advisor-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Clinical Advisor Box</div>', unsafe_allow_html=True)
    advisor_left, advisor_right = st.columns([1.4, 1.0], gap="large")

    with advisor_left:
        st.markdown('<div class="advisor-card">', unsafe_allow_html=True)
        st.markdown('<div class="advisor-title">Claude interpretation</div>', unsafe_allow_html=True)
        if prediction_payload:
            if st.button("Generate Clinical Interpretation", key="groq_advisor_btn"):
                with st.spinner("Generating clinical interpretation..."):
                    advisor_text = call_groq_advisor(prediction_payload)
                    st.session_state["track_b_advisor_text"] = advisor_text
            advisor_text = st.session_state.get("track_b_advisor_text")
            if advisor_text:
                advisor_html = f"""
<div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
             border: 1px solid #00d4ff;
             border-left: 4px solid #00d4ff;
             border-radius: 12px;
             padding: 24px;
             margin-top: 16px;">
    <div style="color: #f5f7fb; line-height: 1.7; white-space: pre-wrap;">{html.escape(advisor_text)}</div>
</div>
"""
                st.markdown(advisor_html, unsafe_allow_html=True)
            else:
                advisor_html = f"""
<div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
             border: 1px solid #00d4ff;
             border-left: 4px solid #00d4ff;
             border-radius: 12px;
             padding: 24px;
             margin-top: 16px;">
    <p style="color: #a0a0b0; font-style: italic;">
        &#128300; Click 'Generate Clinical Interpretation' to get AI-powered antibiotic stewardship recommendations.
    </p>
</div>
"""
                st.markdown(advisor_html, unsafe_allow_html=True)
        else:
            advisor_html = f"""
<div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
             border: 1px solid #00d4ff;
             border-left: 4px solid #00d4ff;
             border-radius: 12px;
             padding: 24px;
             margin-top: 16px;">
    <p style="color: #a0a0b0; font-style: italic;">
        &#128300; Click 'Generate Clinical Interpretation' to get AI-powered antibiotic stewardship recommendations.
    </p>
</div>
"""
            st.markdown(advisor_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with advisor_right:
        st.markdown('<div class="advisor-card">', unsafe_allow_html=True)
        st.markdown('<div class="advisor-title">Key Drivers</div>', unsafe_allow_html=True)
        if prediction_payload:
            for item in local_top_features[:3]:
                st.markdown(
                    f'<span class="pill">{item["feature"]}: {item["value"]:+.4f}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="advisor-title">Stewardship Recs</div>', unsafe_allow_html=True)
            for rec in extract_stewardship_recs(prediction_label, resistant_probability, local_top_features):
                st.markdown(f"- {rec}")
            st.markdown('<div class="advisor-title" style="margin-top:14px;">Note on Uncertainty</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="small-note">Culture validation mandatory. Research grade output only. Clinical correlation required before deployment or treatment decisions.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            st.download_button(
                "Download Structured Report (JSON)",
                data=json.dumps(prediction_payload, indent=2),
                file_name=f"{case_id or 'track_b_report'}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.write("No report yet. Predict first to unlock the structured summary.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
