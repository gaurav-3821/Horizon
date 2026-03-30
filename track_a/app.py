# RUN COMMAND: streamlit run track_a/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
SHAP_SUMMARY_PATH = ARTIFACTS_DIR / "shap_summary.png"
TRAINING_RESULTS_PATH = ARTIFACTS_DIR / "training_results.png"

BACKGROUND = "#f5f5f0"
CARD_BG = "#ffffff"
BORDER = "#000000"
ACCENT = "#0066cc"
TEXT = "#0a0a0f"

st.set_page_config(page_title="Horizon | Track A", layout="wide", page_icon="🧪")

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
    .card {{
        background: {CARD_BG};
        border: 2px solid {ACCENT};
        padding: 20px;
        box-shadow: 4px 4px 0px {ACCENT};
        height: 100%;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 2px solid {BORDER};
        padding: 18px 20px;
        min-height: 122px;
        box-shadow: 4px 4px 0px {BORDER};
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
        margin-bottom: 0.9rem;
    }}
    .info-copy {{
        color: {TEXT};
        font-size: 0.98rem;
        line-height: 1.7;
        margin-bottom: 0.9rem;
    }}
</style>
"""


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
            "🧬 Track B — Antibiotic Resistance",
            "🧪 Track A — Drug Toxicity",
            "🦠 Track C — Epidemic Spread",
        ],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown('<div class="hero-title">Drug Toxicity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Track A — Molecular GNN + Tabular Hybrid</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-banner">&#9888; This track requires local scientific dependencies for full inference. Cloud deployment currently shows a static summary view.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        render_metric_card("&#129514;", "Dataset", "7,831", "Tox21 compounds")
    with m2:
        render_metric_card("&#129516;", "Model", "Hybrid GNN", "Graph + tabular encoder")
    with m3:
        render_metric_card("&#128200;", "Endpoints", "12", "Toxicity tasks predicted")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Track Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-copy">This track uses a hybrid GNN + tabular model trained on Tox21 dataset (7,831 compounds).</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-copy">Model predicts 12 toxicity endpoints from molecular SMILES structure.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-copy">Requires rdkit and torch-geometric — available in local deployment.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">SHAP Summary</div>', unsafe_allow_html=True)
        if SHAP_SUMMARY_PATH.exists():
            st.image(str(SHAP_SUMMARY_PATH), use_container_width=True)
        else:
            st.info("SHAP summary image not found.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Training Results</div>', unsafe_allow_html=True)
        if TRAINING_RESULTS_PATH.exists():
            st.image(str(TRAINING_RESULTS_PATH), use_container_width=True)
        else:
            st.info("Training results image not found.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
