"""The four fundamental subspaces, rank and nullity."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..ui import fmt_vec, header, matrix_editor, show, tip
from ..viz import plot as P

DEFAULTS = {
    (2, 2): [[1, 2], [2, 4]],
    (2, 3): [[1, 2, 1], [2, 4, 2]],
    (3, 2): [[1, 0], [0, 1], [1, 1]],
    (3, 3): [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
}


def subspace_traces(Q: np.ndarray, space_dim: int, span: float, color: str, name: str) -> list:
    """Draw span of the orthonormal columns of Q inside R^space_dim."""
    d = Q.shape[1]
    if d == 0:
        return [P.points_2d(np.zeros(2), color, f"{name}: only 0", 12) if space_dim == 2
                else P.points_3d(np.zeros(3), color, f"{name}: only 0", 7)]
    if d == 1:
        return [P.line_through_origin_2d(Q[:, 0], span, color, f"{name} (line)", "solid", 4) if space_dim == 2
                else P.line_3d(Q[:, 0], span, color, f"{name} (line)", 8, None)]
    if d == 2 and space_dim == 3:
        return [P.plane_through_origin(Q[:, 0], Q[:, 1], span, color, f"{name} (plane)", 0.35)]
    s = span * 0.98
    if space_dim == 2:
        return [P.polygon_2d(np.array([[-s, -s], [s, -s], [s, s], [-s, s]]), color, f"{name}: all of R²", 0.12)]
    return [P.points_3d(np.zeros((1, 3)), color, f"{name}: all of R³", 1)]


def render() -> None:
    header("Subspaces & rank",
           "Every matrix splits space into four pieces: what it keeps, what it kills, what it reaches, what it misses.")
    c = st.columns(3)
    m = c[0].radio("Rows (output dimension m)", [2, 3], index=1, horizontal=True, key="sub_m")
    n = c[1].radio("Columns (input dimension n)", [2, 3], index=1, horizontal=True, key="sub_n")
    span = c[2].slider("View size", 1.5, 10.0, 3.0, 0.5, key="sub_span")
    A = matrix_editor("sub_A", m, n, DEFAULTS[(m, n)], label="A")
    if A is None:
        return
    r = la.rank(A)
    row, null = la.row_space(A), la.null_space(A)
    col, left_null = la.column_space(A), la.left_null_space(A)
    st.markdown(f"### rank = {r}  ·  nullity = {n - r}  ·  rank + nullity = {n} (number of columns)")

    rng = np.random.default_rng(0)
    cloud = rng.uniform(-1, 1, (300, n)) * span * 0.6
    image = cloud @ A.T
    shade = cloud[:, 0]

    def cloud_trace(Pts, dim, name):
        if dim == 2:
            return go.Scatter(x=Pts[:, 0], y=Pts[:, 1], mode="markers", name=name,
                              marker=dict(size=5, color=shade, colorscale="Viridis", opacity=0.75))
        return go.Scatter3d(x=Pts[:, 0], y=Pts[:, 1], z=Pts[:, 2], mode="markers", name=name,
                            marker=dict(size=2.5, color=shade, colorscale="Viridis", opacity=0.75))

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Input space R{'²³'[n - 2]}**")
        tr = [] if n == 2 else [P.axes_3d(span)]
        tr += subspace_traces(row, n, span, "#4dabf7", "row space")
        tr += subspace_traces(null, n, span, "#ff6b6b", "null space")
        tr.append(cloud_trace(cloud, n, "input points"))
        lay = P.layout_2d(span, 520) if n == 2 else P.layout_3d(span, 520)
        show(go.Figure(tr, lay), key="sub_in")
    with right:
        st.markdown(f"**Output space R{'²³'[m - 2]}** — where A sends those points")
        tr = [] if m == 2 else [P.axes_3d(span)]
        tr += subspace_traces(col, m, span, "#51cf66", "column space")
        tr += subspace_traces(left_null, m, span, "#ffd43b", "left null space")
        tr.append(cloud_trace(image, m, "A × points"))
        lay = P.layout_2d(span, 520) if m == 2 else P.layout_3d(span, 520)
        show(go.Figure(tr, lay), key="sub_out")

    cols = st.columns(4)
    info = [("Column space C(A)", col, "#51cf66", "Every output Ax can reach. Dimension = rank."),
            ("Null space N(A)", null, "#ff6b6b", "Inputs crushed to 0 (Ax = 0)."),
            ("Row space C(Aᵀ)", row, "#4dabf7", "Perpendicular to the null space."),
            ("Left null space N(Aᵀ)", left_null, "#ffd43b", "Output directions perpendicular to everything A can produce.")]
    for col_ui, (title, Q, colr, desc) in zip(cols, info):
        with col_ui:
            st.markdown(f"<span style='color:{colr}'>■</span> **{title}** — dim {Q.shape[1]}",
                        unsafe_allow_html=True)
            st.caption(desc)
            for k in range(Q.shape[1]):
                st.code(fmt_vec(Q[:, k]), language=None)
    tip("Colours of the dots match between the plots, so you can follow each point. All dots land inside the "
        "green column space. Moving an input along the red null space does not change its output at all — "
        "that is why a matrix with a non-trivial null space cannot be inverted.")
