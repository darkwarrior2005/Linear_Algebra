"""Linear Algebra Visualizer — entry point.

Run with:   streamlit run app.py      (or:  python run.py)
"""

import streamlit as st

from linalg_viz.registry import build_pages

st.set_page_config(page_title="Linear Algebra Visualizer", page_icon="🧮", layout="wide")
st.navigation(build_pages(), expanded=True).run()
