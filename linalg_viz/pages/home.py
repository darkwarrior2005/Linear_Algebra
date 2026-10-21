"""Welcome page: what is in the app and how to type things."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..presets import MATRICES
from ..registry import PAGES
from ..ui import show
from ..viz.scenes import SceneOptions, linear_path, transform_figure

TOUR = [
    ("➡️ Vectors", "vectors", "Add, scale and combine arrows; see span, dot and cross products."),
    ("🔀 Linear transformations", "transformations", "Watch a matrix bend the grid. Columns = where the basis lands."),
    ("🔗 Composition & inverse", "composition", "Multiplication is one move after another; inverses undo."),
    ("🎯 Systems Ax = b", "systems", "Lines/planes crossing, columns combining, row reduction step by step."),
    ("🧊 Subspaces & rank", "subspaces", "Column space, null space, row space and rank–nullity."),
    ("🧭 Change of basis", "change-of-basis", "Same arrow, different rulers; the same map in a new basis."),
    ("📏 Gram–Schmidt & projections", "gram-schmidt", "Make vectors perpendicular; cast shadows on subspaces."),
    ("🌟 Eigenvectors & eigenvalues", "eigen", "The directions a matrix only stretches."),
    ("🧩 Decompositions", "decompositions", "SVD, eigen, polar, QR, LU as animated sequences of simple moves."),
    ("🧮 Matrix calculator", "calculator", "Any size: inverse, RREF, eigen, SVD, null space, exp(A)…"),
    ("🌬️ Vector fields", "vector-fields", "Type F(x, y, z, t) — arrows, streamlines, time animation."),
    ("🧲 Differential operators", "operators", "Gradient, divergence, curl, Laplacian, Jacobian — with probes."),
    ("🌊 Flows & particles", "flows", "Release particles or a blob of fluid and let time run."),
    ("🌀 Linear dynamics x′ = Ax", "linear-dynamics", "Eigenvalues decide spirals, saddles and stability."),
    ("🛰️ Motion along a curve r(t)", "motion", "Velocity, acceleration and the moving T-N-B frame."),
]


def render() -> None:
    st.title("🧮 Linear Algebra Visualizer")
    st.markdown("#### See it, move it, change it — linear algebra and vector calculus in 2D and 3D.")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("Every page is a playground: type a matrix, a vector or a formula and the picture updates. "
                    "Press **▶ Play** on any plot to animate it, drag to rotate 3D views, scroll to zoom.")
        cols = st.columns(2)
        for k, (title, url, desc) in enumerate(TOUR):
            with cols[k % 2].container(border=True):
                label = f"**{title.split(' ', 1)[1]}**"
                if url in PAGES:
                    st.page_link(PAGES[url], label=label)
                else:  # page registry not built (e.g. the page is rendered on its own)
                    st.markdown(label)
                st.caption(desc)
    with right:
        A = np.array(MATRICES[2]["Symmetric (eigen-friendly)"], dtype=float)
        show(transform_figure(2, linear_path(A), np.linspace(0, 1, 30),
                              SceneOptions(span=4, circle=True, height=460), initial=0), key="home_demo")
        with st.expander("✍️ How to type numbers and formulas", expanded=True):
            st.markdown(
                "- Matrix cells accept numbers **or expressions**: `sqrt(2)/2`, `cos(pi/6)`, `-1/3`, `2e-3`.\n"
                "- Field formulas use `x, y, z` and time `t`: `-y + 0.1*x`, `sin(x)*exp(-t)`, `x^2 - y^2`.\n"
                "- Available: `sin cos tan asin acos atan atan2 sinh cosh tanh exp log ln sqrt abs sign "
                "floor ceil min max where(cond, a, b) pi e`.\n"
                "- `^` means power, and `2x` means `2*x`.")
        with st.expander("🎨 Colour code"):
            st.markdown("<span style='color:#ff6b6b'>■</span> î / x-direction &nbsp; "
                        "<span style='color:#51cf66'>■</span> ĵ / y-direction &nbsp; "
                        "<span style='color:#4dabf7'>■</span> k̂ / z-direction &nbsp; "
                        "<span style='color:#ffd43b'>■</span> eigen-things & highlights",
                        unsafe_allow_html=True)


