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
    "Track B — Antibiotic Resistance": PROJECT_ROOT / "track_b" / "app.py",
    "Track A — Drug Toxicity": PROJECT_ROOT / "track_a" / "app.py",
    "Track C — Epidemic Spread": PROJECT_ROOT / "track_c" / "app.py",
}
DEFAULT_TRACK = "Track B — Antibiotic Resistance"


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
            [data-testid="stSidebar"] {{display: none !important;}}
            .block-container {{
                max-width: 1800px;
                padding-top: 1.0rem;
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
            .drawer-card {{
                background: #ffffff;
                border: 2px solid {BORDER};
                box-shadow: 4px 4px 0px {BORDER};
                padding: 18px;
                margin-bottom: 1rem;
            }}
            .drawer-heading {{
                font-size: 1rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.85rem;
                color: {TEXT};
            }}
            .drawer-note {{
                color: {TEXT};
                font-size: 0.9rem;
                line-height: 1.5;
                margin-bottom: 1rem;
            }}
            .status-card {{
                background: #ffffff;
                border: 2px solid {BORDER};
                box-shadow: 4px 4px 0px {BORDER};
                padding: 14px 16px;
                margin-bottom: 1rem;
            }}
            .status-title {{
                font-size: 0.9rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: {TEXT};
                margin-bottom: 0.5rem;
            }}
            .status-line {{
                color: {TEXT};
                font-size: 0.88rem;
                line-height: 1.5;
                margin-bottom: 0.3rem;
            }}
            .track-pill {{
                display: inline-block;
                margin-bottom: 0.75rem;
                padding: 6px 10px;
                border: 2px solid {ACCENT};
                box-shadow: 3px 3px 0px {ACCENT};
                font-weight: 700;
                color: {TEXT};
                background: #eef5ff;
            }}
            .stButton > button {{
                background: #ffffff;
                color: {TEXT};
                border: 2px solid {BORDER} !important;
                border-radius: 0px !important;
                box-shadow: 4px 4px 0px {BORDER};
                font-weight: 800;
            }}
            .stButton > button:hover {{
                background: {ACCENT};
                color: #ffffff;
                box-shadow: 2px 2px 0px {BORDER};
                transform: translate(2px, 2px);
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


def render_drawer():
    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">Prototype Status</div>
            <div class="status-line">Track B is the flagship live demo.</div>
            <div class="status-line">Track A is rebuilding on tabular-only models.</div>
            <div class="status-line">Track C covers epidemic forecasting.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="drawer-card">', unsafe_allow_html=True)
    st.markdown('<div class="drawer-heading">Tracks</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="track-pill">Current: {st.session_state["selected_track"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="drawer-note">Choose a track to switch between Antibiotic Resistance, Drug Toxicity, and Epidemic Spread modules.</div>',
        unsafe_allow_html=True,
    )

    for track_name in TRACKS:
        if st.button(track_name, use_container_width=True, key=f"nav_{track_name}"):
            st.session_state["selected_track"] = track_name
            st.session_state["nav_open"] = False
            st.rerun()

    if st.button("Close Navigation", use_container_width=True, key="nav_close"):
        st.session_state["nav_open"] = False
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Horizon", layout="wide")
    inject_css()

    if "selected_track" not in st.session_state:
        st.session_state["selected_track"] = DEFAULT_TRACK
    if "nav_open" not in st.session_state:
        st.session_state["nav_open"] = False

    top_left, top_right = st.columns([0.18, 0.82], gap="small")
    with top_left:
        button_label = "Hide Tracks" if st.session_state["nav_open"] else "Show Tracks"
        if st.button(button_label, use_container_width=True, key="toggle_drawer"):
            st.session_state["nav_open"] = not st.session_state["nav_open"]
            st.rerun()
    with top_right:
        st.markdown(
            """
            <div class="hub-topbar">
                <div>
                    <div class="hub-title">Horizon</div>
                    <div class="hub-subtitle">CodeCure AI Hackathon</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state["nav_open"]:
        nav_col, main_col = st.columns([0.24, 0.76], gap="large")
        with nav_col:
            render_drawer()
        with main_col:
            run_track(st.session_state["selected_track"], TRACKS[st.session_state["selected_track"]])
    else:
        run_track(st.session_state["selected_track"], TRACKS[st.session_state["selected_track"]])


if __name__ == "__main__":
    main()
