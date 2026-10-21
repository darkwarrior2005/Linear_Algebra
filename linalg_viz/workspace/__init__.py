"""The single-graph workspace: controls in the sidebar, one big graph, legend + info underneath."""

from __future__ import annotations

import streamlit as st

from ..ui import show
from . import calculator, panels
from .controls import read_controls
from .scene import Scene

PAGE_CSS = """
<style>
section[data-testid="stSidebar"] {width: 380px !important;}
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%;}
</style>
"""

PLAY_OPTIONS = ["auto", "the matrix", "time"]


def _time_layers_on(s) -> bool:
    field_moves = s.field_on and s.F is not None and s.F.time_dependent
    probe_moves = s.probe_on and "paddle wheel (curl)" in s.probe_extras and s.F is not None
    return bool(s.particles_on or s.curve_on or field_moves or probe_moves or s.space_flow)


def render() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    s = read_controls()
    top = st.columns([3, 2])
    with top[0]:
        choice = st.segmented_control("▶ Play animates", PLAY_OPTIONS, default="auto", key="ws_play") or "auto"
    has_matrix = s.tf_on and bool(s.steps)
    if choice == "auto":  # things that move in time win; otherwise play the matrix
        mode = "time" if (_time_layers_on(s) or not has_matrix) else "transform"
    else:
        mode = "transform" if choice == "the matrix" else "time"
    with top[1]:
        st.caption("Drag to pan/rotate · scroll to zoom · double-click to reset the view. "
                   + ("▶ applies the matrix step by step." if mode == "transform" else
                      f"▶ runs time from {s.t0:g} to {s.t1:g}."))
    for msg in s.errors:
        st.warning(msg)
    scene = Scene(s)
    try:
        fig = scene.build(mode)
    except Exception as exc:  # keep the app usable if one layer cannot be drawn
        st.error(f"Could not draw the graph: {exc}")
        return
    show(fig, key="ws_graph")
    panels.legend(scene.keys)
    panels.info(s, scene)
    with st.expander("🧮 Matrix calculator — any size (inverse, RREF, eigen, SVD, null space, eᴬ, solve…)"):
        calculator.render()
