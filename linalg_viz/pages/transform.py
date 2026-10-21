"""Linear transformations: watch a matrix move all of space."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..core import linalg as la
from ..presets import MATRICES
from ..ui import dimension_picker, fmt, fmt_vec, header, matrix_editor, show, tip, vector_editor
from ..viz import plot as P
from ..viz.scenes import SceneOptions, linear_path, transform_figure


def render() -> None:
    header("Linear transformations",
           "A matrix is a machine that moves every point of space. Press ▶ Play to watch it happen.")
    dim = dimension_picker("tf_dim")
    left, right = st.columns([1, 2.3])
    with left:
        default = MATRICES[dim]["Shear (horizontal)" if dim == 2 else "Shear x by z"]
        A = matrix_editor("tf_A", dim, dim, default, MATRICES[dim], label="A")
        with st.expander("What to show", expanded=True):
            grid = st.checkbox("Grid lines", True, key="tf_grid")
            shape = st.checkbox("Unit square / cube", True, key="tf_shape")
            circle = st.checkbox("Unit circle / sphere", dim == 2, key=f"tf_circle{dim}")
            eig = st.checkbox("Eigenvector lines", False, key="tf_eig")
            fan = st.checkbox("Fan of unit vectors", False, key="tf_fan") if dim == 2 else False
            use_v = st.checkbox("A vector of my own", True, key="tf_usev")
            v = vector_editor("tf_v", dim, [1, 2, 1][:dim], "v", P.color(2)) if use_v else None
            span = st.slider("View size", 1.5, 10.0, 4.0 if dim == 2 else 2.5, 0.5, key=f"tf_span{dim}")
            frames = st.slider("Animation frames", 10, 60, 30, 5, key="tf_frames")
    if A is None:
        return
    vectors = []
    if v is not None:
        vectors.append((v, P.color(2), "v → Av"))
    lines = []
    if eig:
        for p in la.eigen_analysis(A):
            if p.is_real:
                lines.append((np.real(p.vector), P.EIGEN_COLOR, f"eigenline λ = {fmt(p.value.real)}"))
    o = SceneOptions(span=span, grid=grid, shape=shape, circle=circle, fan=24 if fan else 0,
                     vectors=vectors, lines=lines)
    ts = np.linspace(0, 1, frames)
    fig = transform_figure(dim, linear_path(A), ts, o)
    with right:
        show(fig, key="tf_fig")

    det = float(np.linalg.det(A))
    c = st.columns(4)
    c[0].metric("determinant", fmt(det))
    c[1].metric("rank", la.rank(A))
    c[2].metric("trace", fmt(np.trace(A)))
    c[3].metric("invertible?", "yes" if abs(det) > 1e-9 else "no")
    cols = "î, ĵ" if dim == 2 else "î, ĵ, k̂"
    st.markdown(f"**Columns of A are where {cols} land:** "
                + ", ".join(f"{lab} → {fmt_vec(A[:, k])}" for k, lab in enumerate(["î", "ĵ", "k̂"][:dim])))
    if v is not None:
        st.markdown(f"**Your vector:** v = {fmt_vec(v)} → Av = {fmt_vec(A @ v)}  "
                    f"(= " + " + ".join(f"{fmt(v[k])}·{lab}'" for k, lab in enumerate(['î', 'ĵ', 'k̂'][:dim]))
                    + ")")
    for fact in la.describe_transform(A):
        st.markdown(f"- {fact}")
    tip("Grid lines stay straight, parallel and evenly spaced, and the origin stays put — that is what "
        "makes a transformation *linear*. Because of that, knowing where the basis arrows land tells you "
        "where every point lands: Av = v₁·(first column) + v₂·(second column) + …")
