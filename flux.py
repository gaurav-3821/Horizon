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
    "Track A - Drug Toxicity": PROJECT_ROOT / "track_a" / "app.py",
    "Track B - Antibiotic Resistance": PROJECT_ROOT / "track_b" / "app.py",
    "Track C - Epidemic Spread": PROJECT_ROOT / "track_c" / "app.py",
}
DEFAULT_TRACK = "Track A - Drug Toxicity"


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
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {{
                background-color: {BACKGROUND} !important;
                background-image: 
                    linear-gradient(to right, #cccccc 1px, transparent 1px),
                    linear-gradient(to bottom, #cccccc 1px, transparent 1px) !important;
                background-size: 40px 40px !important;
                color: {TEXT} !important;
            }}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
            [data-testid="stHeader"] {{display: none;}}
            [data-testid="stSidebar"] {{display: none !important;}}
            .block-container {{
                max-width: 1800px;
                padding-top: 100px; /* Space for the fixed header */
                padding-bottom: 1.25rem;
            }}
            .hub-topbar {{
                display: flex;
                align-items: center;
                gap: 14px;
                margin-bottom: 1rem;
            }}
            .hub-title {{
                font-size: 2rem;
                font-weight: 800;
                color: {TEXT};
                line-height: 1.1;
            }}
            .hub-subtitle {{
                color: {TEXT};
                font-size: 0.95rem;
                margin-top: 2px;
            }}
            div[data-testid="stVerticalBlock"]:has(> div.element-container .top-nav-hook) {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                background-color: {BACKGROUND};
                z-index: 99999;
                padding: 2rem 3rem 1rem 3rem;
                border-bottom: 3px solid #000000;
                margin-top: 0;
            }}
            .stButton > button {{
                cursor: pointer;
                height: 3rem !important;
                border: 2px solid black !important;
                padding: 0.625rem !important;
                background-color: #A6FAFF !important;
                transition: all 0.1s ease-in-out;
                user-select: none;
                border-radius: 0.375rem !important;
                box-shadow: none !important;
                font-weight: 600;
                font-size: 1.05em;
                color: {TEXT} !important;
            }}
            .stButton > button:hover {{
                background-color: #79F7FF !important;
                box-shadow: 2px 2px 0px rgba(0,0,0,1) !important;
                transform: translate(-1px, -1px);
            }}
            .stButton > button:active {{
                background-color: #00E1EF !important;
                box-shadow: 0px 0px 0px rgba(0,0,0,1) !important;
                transform: translate(0px, 0px);
            }}

            /* Selectbox styling */
            div[data-baseweb="select"] > div {{
                background-color: #B8FF9F !important;
                border: 2px solid black !important;
                border-radius: 0 !important;
                transition: all 0.1s ease-in-out;
            }}
            div[data-baseweb="select"] > div:hover {{
                background-color: #99fc77 !important;
            }}
            div[data-baseweb="select"] > div:focus-within {{
                box-shadow: 2px 2px 0px rgba(0,0,0,1) !important;
            }}
            div[data-baseweb="select"] span {{
                color: black !important;
                font-weight: 500 !important;
            }}
            
            /* Dropdown popover styling */
            div[data-baseweb="popover"] > div {{
                border: 2px solid black !important;
                box-shadow: 2px 2px 0px rgba(0,0,0,1) !important;
                border-radius: 0 !important;
                background-color: white !important;
            }}
            div[data-baseweb="popover"] ul {{
                background-color: white !important;
                padding: 0 !important;
            }}
            div[data-baseweb="popover"] li {{
                border-bottom: 2px solid black !important;
                color: black !important;
                font-size: 0.875rem !important;
                padding: 0.5rem 1rem !important;
            }}
            div[data-baseweb="popover"] li:last-child {{
                border-bottom: none !important;
            }}
            div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {{
                background-color: #B8FF9F !important;
                font-weight: 600 !important;
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
        if track_name == "Track A — Drug Toxicity":
            st.warning("Track A models are currently compiling. Please check back later.")
        else:
            st.error(f"{track_name} is missing required files.")
    except Exception as exc:
        if track_name == "Track A — Drug Toxicity":
            st.warning("Track A models are currently compiling. Please check back later.")
        else:
            st.error(f"{track_name} failed to load: {exc}")
            st.exception(exc)
    finally:
        sys.path[:] = original_sys_path
        st.set_page_config = original_set_page_config


def render_top_nav():
    track_list = list(TRACKS.keys())
    with st.container():
        st.markdown('<div class="top-nav-hook"></div>', unsafe_allow_html=True)
        cols = st.columns([1.5, 1, 1, 1], gap="small")
        with cols[0]:
            st.markdown(
                """
                <div class="hub-topbar" style="margin-bottom: 0px;">
                    <div>
                        <div class="hub-title" style="margin-top: -6px;">Horizon</div>
                        <div class="hub-subtitle">CodeCure AI Hackathon</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for i, track_name in enumerate(track_list):
            with cols[i + 1]:
                if st.button(track_name, use_container_width=True, key=f"nav_{track_name}"):
                    st.session_state["selected_track"] = track_name
                    st.rerun()


def main():
    st.set_page_config(page_title="Horizon", layout="wide")
    inject_css()

    if "selected_track" not in st.session_state:
        st.session_state["selected_track"] = DEFAULT_TRACK

    render_top_nav()
    run_track(st.session_state["selected_track"], TRACKS[st.session_state["selected_track"]])


if __name__ == "__main__":
    main()
