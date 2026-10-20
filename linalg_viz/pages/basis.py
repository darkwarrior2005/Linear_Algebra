"""Change of basis: the same arrow described in different coordinate systems."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..presets import MATRICES
from ..ui import (dimension_picker, fmt_vec, header, latex_matrix, matrix_editor, set_matrix, show, tip,
                  vector_editor)
from ..viz import plot as P
from ..viz.scenes import SceneOptions, linear_path, transform_figure

BASIS_DEFAULT = {2: [[2, -1], [1, 1]], 3: [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}


def _coordinates(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        st.markdown("Basis vectors are the **columns**.")
        B = matrix_editor("cob_B", dim, dim, BASIS_DEFAULT[dim], label="Basis B = [b₁ b₂ …]")
        mode = st.radio("I will type v in…", ["standard coordinates", "B-coordinates"], key="cob_mode")
        v_in = vector_editor("cob_v", dim, [3, 1, 2][:dim], "v", P.color(0))
        span = st.slider("View size", 1.5, 10.0, 4.0, 0.5, key=f"cob_span{dim}")
    if B is None or v_in is None:
        return
    if abs(np.linalg.det(B)) < 1e-10:
        st.error("These vectors are not a basis: they are linearly dependent (det = 0).")
        return
    if mode == "standard coordinates":
        v, c = v_in, la.change_of_basis(v_in, B)
    else:
        c, v = v_in, B @ v_in
    head = span * (0.045 if dim == 2 else 0.07)
    tr: list = []
    if dim == 2:
        tr.append(P.grid_2d(np.eye(2), 8, color=P.GRID_FAINT, width=1))
        tr.append(P.grid_2d(B, 8, color="rgba(255,146,43,0.45)", width=1.3, name="B-grid", showlegend=True))
        arrows = P.arrows_2d
    else:
        tr.append(P.axes_3d(span))
        tr.append(P.grid_3d(B, 2, 1, color="rgba(255,146,43,0.35)", width=2, name="B-grid"))
        arrows = P.arrows_3d
    origin = np.zeros(dim)
    for k in range(dim):
        tr += arrows(B[:, k], color=P.BASIS_COLORS[k], name=f"b{k + 1}", head=head, labels=[f"b{k + 1}"])
        tr += arrows(c[k] * B[:, k], origin, color=P.BASIS_COLORS[k], name=f"{c[k]:.2f}·b{k + 1}", head=head,
                     opacity=0.5)
        origin = origin + c[k] * B[:, k]
    tr += arrows(v, color=P.color(0), name="v", head=head, width=5)
    lay = P.layout_2d(span, 600) if dim == 2 else P.layout_3d(span, 600)
    with right:
        show(go.Figure(tr, lay), key="cob_fig")
    st.markdown(f"**Standard coordinates:** v = {fmt_vec(v)}  ·  **B-coordinates:** [v]_B = {fmt_vec(c)}")
    st.latex(r"v = B\,[v]_B \qquad [v]_B = B^{-1} v")
    tip("The arrow never moves — only the ruler changes. The orange grid is the coordinate system made by "
        "b₁, b₂, …; [v]_B counts how many steps of each basis vector you take to reach v.")


def _similar(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        A = matrix_editor("sim_A", dim, dim, MATRICES[dim]["Symmetric (eigen-friendly)" if dim == 2 else "Symmetric"],
                          MATRICES[dim], label="A (in standard coordinates)")
        B = matrix_editor("sim_B", dim, dim, BASIS_DEFAULT[dim], label="Basis B (columns)")
        if A is not None:
            vals, vecs = np.linalg.eig(A)
            real = np.allclose(vals.imag, 0) and abs(np.linalg.det(vecs)) > 1e-8
            st.button("Use the eigenvectors of A as the basis", disabled=not real,
                      on_click=lambda: set_matrix("sim_B", np.real(vecs) / np.max(np.abs(np.real(vecs)), axis=0)),
                      help="In the eigenbasis the matrix becomes diagonal: pure stretching along each axis.")
        span = st.slider("View size", 1.5, 10.0, 4.0 if dim == 2 else 2.5, 0.5, key=f"sim_span{dim}")
    if A is None or B is None:
        return
    if abs(np.linalg.det(B)) < 1e-10:
        st.error("B is not a basis (det = 0).")
        return
    AB = np.linalg.inv(B) @ A @ B
    with right:
        a, b = st.columns(2)
        o = SceneOptions(span=span, height=430, circle=dim == 2)
        with a:
            st.markdown("**A** acting in standard coordinates")
            show(transform_figure(dim, linear_path(A), np.linspace(0, 1, 21), o), key="sim_f1")
        with b:
            st.markdown("**B⁻¹AB** — the same map in B's coordinates")
            show(transform_figure(dim, linear_path(AB), np.linspace(0, 1, 21), o), key="sim_f2")
    latex_matrix(AB, r"[A]_B = B^{-1} A B = ")
    if np.allclose(AB, np.diag(np.diag(AB)), atol=1e-8):
        st.success("In this basis the transformation is diagonal: it only stretches along the basis directions.")
    tip("Same transformation, different description. Determinant, trace and eigenvalues are the same in every "
        "basis — they belong to the transformation, not to the coordinates.")


def render() -> None:
    header("Change of basis", "One arrow, many coordinate systems.")
    dim = dimension_picker("cob_dim")
    t1, t2 = st.tabs(["🧭 Coordinates in a new basis", "🔄 A transformation in a new basis"])
    with t1:
        _coordinates(dim)
    with t2:
        _similar(dim)
