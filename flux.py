# RUN COMMAND: streamlit run flux.py
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_script_path(*candidates: str) -> Path | None:
    for rel in candidates:
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    return None


def build_page(title: str, icon: str | None, url_path: str, *candidates: str):
    script_path = resolve_script_path(*candidates)
    if script_path is None:
        st.sidebar.warning(f"Missing page script for {title}: {', '.join(candidates)}")
        return None
    return st.Page(str(script_path), title=title, icon=icon, url_path=url_path)


def main():
    st.set_page_config(page_title="Horizon", layout="wide")

    if not hasattr(st, "Page") or not hasattr(st, "navigation"):
        st.error("This Streamlit version does not support st.Page/st.navigation.")
        return

    st.sidebar.title("Horizon")
    st.sidebar.caption("Unified navigation across all three ML tracks.")

    pages = [
        build_page("Track A: Toxicity", "\U0001F9EC", "track-a", "track_a/app.py"),
        build_page("Track B: Resistance", "\U0001F9A0", "track-b", "track_b/app.py"),
        build_page("Track C: Epidemic", "\U0001F4C8", "track-c", "track_c/app.py"),
    ]
    pages = [page for page in pages if page is not None]

    if not pages:
        st.error("No valid pages found. Check track_a/app.py, track_b/app.py, and track_c/app.py.")
        return

    try:
        nav = st.navigation(pages, position="sidebar")
    except TypeError:
        nav = st.navigation(pages)

    try:
        nav.run()
    except Exception as exc:
        st.warning("Selected page failed to load. Choose another page from the sidebar.")
        st.exception(exc)


if __name__ == "__main__":
    main()
