# RUN COMMAND: streamlit run hub/master_app.py
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_script_path(*candidates: str) -> Path | None:
    for rel in candidates:
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    return None


def build_page(title: str, icon: str | None, *candidates: str):
    script_path = resolve_script_path(*candidates)
    if script_path is None:
        st.sidebar.warning(f"Missing page script for {title}: {', '.join(candidates)}")
        return None
    return st.Page(str(script_path), title=title, icon=icon)


def main():
    st.sidebar.title("Model Test Hub")
    st.sidebar.caption("Use the page links below to test each track independently.")

    pages = [
        build_page("Track A: Toxicity", "🧬", "track_a/app.py"),
        build_page("Track B: Resistance", "🦠", "track_b/app.py"),
        build_page("Track C: Epidemic", "📈", "track_c/app.py"),
    ]
    pages = [page for page in pages if page is not None]

    if not pages:
        st.error("No valid sub-pages found. Check track_a/app.py, track_b/app.py, and track_c/app.py.")
        return

    try:
        navigator = st.navigation(pages, position="sidebar")
    except TypeError:
        navigator = st.navigation(pages)

    try:
        navigator.run()
    except Exception as exc:
        st.warning("The selected sub-page failed to load. Switch to another page from the sidebar.")
        st.exception(exc)


if __name__ == "__main__":
    main()
