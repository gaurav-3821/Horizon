# RUN COMMAND: streamlit run flux.py
from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
TRACKS = {
    "\U0001F9EC Track B \u2014 Antibiotic Resistance": PROJECT_ROOT / "track_b" / "app.py",
    "\U0001F9EA Track A \u2014 Drug Toxicity": PROJECT_ROOT / "track_a" / "app.py",
    "\U0001F9A0 Track C \u2014 Epidemic Spread": PROJECT_ROOT / "track_c" / "app.py",
}
DEFAULT_TRACK = "\U0001F9EC Track B \u2014 Antibiotic Resistance"


def _noop_set_page_config(*args, **kwargs):
    return None


def run_track(track_name: str, script_path: Path):
    if not script_path.exists():
        st.error(f"Missing page script: {script_path}")
        return

    original_set_page_config = st.set_page_config
    st.set_page_config = _noop_set_page_config
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as e:
        if track_name == "\U0001F9EA Track A \u2014 Drug Toxicity":
            st.error("\u26A0\uFE0F Track A requires rdkit which is unavailable in this environment.")
        elif track_name == "\U0001F9EC Track B \u2014 Antibiotic Resistance":
            st.error(f"Track B failed to load: {str(e)}")
            st.exception(e)
        else:
            st.error("\u26A0\uFE0F This track failed to load. Please check dependencies.")
    finally:
        st.set_page_config = original_set_page_config


def main():
    st.set_page_config(page_title="Horizon", layout="wide", page_icon="\U0001FA7A")

    if "selected_track" not in st.session_state:
        st.session_state["selected_track"] = DEFAULT_TRACK

    st.sidebar.title("Horizon")
    st.sidebar.caption("CodeCure AI Hackathon")
    selected_track = st.sidebar.radio(
        "Tracks",
        options=list(TRACKS.keys()),
        index=list(TRACKS.keys()).index(st.session_state["selected_track"]),
        key="selected_track",
    )

    if selected_track != st.session_state["selected_track"]:
        st.session_state["selected_track"] = selected_track

    run_track(selected_track, TRACKS[selected_track])


if __name__ == "__main__":
    main()
