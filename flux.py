# RUN COMMAND: streamlit run flux.py
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
BACKGROUND = "#f5f5f0"
ACCENT = "#0066cc"
BORDER = "#000000"
TEXT = "#0a0a0f"

TRACKS = {
    "Track B: Antibiotic Resistance": PROJECT_ROOT / "track_b" / "app.py",
    "Track C: Epidemic Forecasting": PROJECT_ROOT / "track_c" / "app.py",
    "Track A: Drug Toxicity": PROJECT_ROOT / "track_a" / "app.py",
}
DEFAULT_TRACK = "Track B: Antibiotic Resistance"


def _noop_set_page_config(*args, **kwargs):
    return None


def inject_css():
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
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
            [data-testid="stHeader"] {{display: none;}}
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
            div[role="radiogroup"] {{
                gap: 0.45rem;
            }}
            div[role="radiogroup"] label {{
                border: 2px solid {BORDER} !important;
                background: #ffffff !important;
                color: {TEXT} !important;
                border-radius: 0px !important;
                box-shadow: 4px 4px 0px {BORDER};
                padding: 0.35rem 0.5rem !important;
                margin-bottom: 0.25rem;
            }}
            div[role="radiogroup"] label:has(input:checked) {{
                background: {ACCENT} !important;
                color: #ffffff !important;
                box-shadow: 4px 4px 0px {ACCENT};
                border-color: {BORDER} !important;
            }}
            div[role="radiogroup"] label span {{
                color: inherit !important;
                font-weight: 700 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_track(track_name: str, script_path: Path):
    if not script_path.exists():
        st.error(f"Missing page script: {script_path}")
        return

    original_set_page_config = st.set_page_config
    original_sys_path = list(sys.path)
    st.set_page_config = _noop_set_page_config
    sys.path.insert(0, str(script_path.parent))
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except FileNotFoundError:
        if track_name == "Track A: Drug Toxicity":
            st.warning("Track A models are currently compiling. Please check back later.")
        else:
            st.error("This track failed to load because a required file is missing.")
    except Exception as exc:
        if track_name == "Track A: Drug Toxicity":
            st.warning("Track A models are currently compiling. Please check back later.")
        else:
            st.error(f"{track_name} failed to load: {exc}")
            st.exception(exc)
    finally:
        sys.path[:] = original_sys_path
        st.set_page_config = original_set_page_config


def main():
    st.set_page_config(page_title="Horizon", layout="wide")
    inject_css()

    if "selected_track" not in st.session_state:
        st.session_state["selected_track"] = DEFAULT_TRACK

    st.sidebar.markdown(
        f"""
        <div style="font-size:1.55rem;font-weight:800;color:{TEXT};margin-bottom:0.1rem;">Horizon</div>
        <div style="color:{TEXT};font-size:0.9rem;margin-bottom:1rem;">CodeCure AI Hackathon</div>
        """,
        unsafe_allow_html=True,
    )

    selected_track = st.sidebar.radio(
        "Tracks",
        options=list(TRACKS.keys()),
        index=list(TRACKS.keys()).index(st.session_state["selected_track"]),
        key="selected_track",
        label_visibility="collapsed",
    )
    st.session_state["selected_track"] = selected_track

    run_track(selected_track, TRACKS[selected_track])


if __name__ == "__main__":
    main()
