"""Matrix multiplication as doing one transformation after another; inverse and transpose."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..core import linalg as la
from ..presets import MATRICES
from ..ui import dimension_picker, fmt, header, latex_matrix, matrix_editor, show, tip
from ..viz import plot as P
from ..viz.scenes import SceneOptions, factor_path, transform_figure


def _options(dim: int, span: float) -> SceneOptions:
    return SceneOptions(span=span, grid=True, shape=True, circle=dim == 2,
                        vectors=[(np.array([1.0, 2.0, 1.0][:dim]), P.color(2), "v")])


def _phase_label(names):
    def label(t: float) -> str:
        k = min(int(t), len(names) - 1) if t < len(names) else len(names) - 1
        return f"{t:.2f}  ({names[k]})"
    return label


def _composition(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        st.markdown("First apply **B**, then **A**. The result is the single matrix **AB**.")
        B = matrix_editor("cmp_B", dim, dim, MATRICES[dim]["Shear (horizontal)" if dim == 2 else "Shear x by z"],
                          MATRICES[dim], label="B (first)")
        A = matrix_editor("cmp_A", dim, dim, MATRICES[dim]["Rotation 90°" if dim == 2 else "Rotate 90° about x"],
                          MATRICES[dim], label="A (second)")
        swap = st.toggle("Swap the order (A first, then B)", key="cmp_swap")
        span = st.slider("View size", 1.5, 10.0, 4.0 if dim == 2 else 2.5, 0.5, key=f"cmp_span{dim}")
    if A is None or B is None:
        return
    first, second, names = (A, B, ["A", "B"]) if swap else (B, A, ["B", "A"])
    product = second @ first
    ts = np.linspace(0, 2, 41)
    fig = transform_figure(dim, factor_path([first, second], rotation_aware=False), ts, _options(dim, span),
                           label=_phase_label([f"applying {names[0]}", f"applying {names[1]}"]))
    with right:
        show(fig, key="cmp_fig")
    c1, c2, c3 = st.columns(3)
    with c1:
        latex_matrix(product, f"{names[1]}{names[0]} = ")
    with c2:
        other = first @ second
        latex_matrix(other, f"{names[0]}{names[1]} = ")
    with c3:
        same = np.allclose(product, other)
        st.markdown("**Order matters?** " + ("Here AB = BA — these two commute." if same else
                                             "Yes! AB ≠ BA: doing them in the other order lands somewhere else."))
        st.markdown(f"det({names[1]}{names[0]}) = det({names[1]})·det({names[0]}) = "
                    f"{fmt(np.linalg.det(second))} × {fmt(np.linalg.det(first))} = {fmt(np.linalg.det(product))}")
    tip("Read matrix products right-to-left, like functions: (AB)v = A(Bv). Each column of AB is "
        "A applied to the matching column of B — i.e. where B sent î, A then sends somewhere else.")


def _inverse(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        A = matrix_editor("inv_A", dim, dim, MATRICES[dim]["Stretch along eigenvectors" if dim == 2 else "Upper triangular"],
                          MATRICES[dim], label="A")
        span = st.slider("View size", 1.5, 10.0, 4.0 if dim == 2 else 2.5, 0.5, key=f"inv_span{dim}")
    if A is None:
        return
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        with right:
            fig = transform_figure(dim, factor_path([A], rotation_aware=False), np.linspace(0, 1, 21),
                                   _options(dim, span))
            show(fig, key="inv_fig_sing")
        st.error(f"det(A) = 0: A squashes space down to rank {la.rank(A)}. Many points land on the same spot, "
                 "so there is no way to undo it — no inverse exists.")
        return
    Ainv = np.linalg.inv(A)
    fig = transform_figure(dim, factor_path([A, Ainv], rotation_aware=False), np.linspace(0, 2, 41),
                           _options(dim, span), label=_phase_label(["applying A", "applying A⁻¹ (undo)"]))
    with right:
        show(fig, key="inv_fig")
    c1, c2 = st.columns(2)
    with c1:
        latex_matrix(Ainv, r"A^{-1} = ")
    with c2:
        latex_matrix(Ainv @ A, r"A^{-1}A = ")
    st.markdown(f"det(A⁻¹) = 1 / det(A) = {fmt(1 / det)}")
    tip("The inverse is the transformation that puts everything back. Play the animation: the second "
        "half runs A⁻¹ and the grid returns exactly to where it started.")


def _transpose(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        A = matrix_editor("tr_A", dim, dim, MATRICES[dim]["Shear (horizontal)" if dim == 2 else "Shear x by z"],
                          MATRICES[dim], label="A")
        span = st.slider("View size", 1.5, 10.0, 4.0 if dim == 2 else 2.5, 0.5, key=f"tr_span{dim}")
    if A is None:
        return
    with right:
        a, b = st.columns(2)
        with a:
            st.markdown("**A**")
            o = _options(dim, span)
            o.height = 430
            show(transform_figure(dim, factor_path([A], rotation_aware=False), np.linspace(0, 1, 21), o),
                 key="tr_figA")
        with b:
            st.markdown("**Aᵀ** (rows ↔ columns)")
            show(transform_figure(dim, factor_path([A.T], rotation_aware=False), np.linspace(0, 1, 21), o),
                 key="tr_figAT")
    latex_matrix(A.T, r"A^T = ")
    st.markdown(f"det(Aᵀ) = det(A) = {fmt(np.linalg.det(A))}. A symmetric matrix (A = Aᵀ) only stretches along "
                "perpendicular directions; an orthogonal matrix has Aᵀ = A⁻¹.")


def render() -> None:
    header("Composition, inverse & transpose",
           "Matrix multiplication = doing one transformation after another.")
    dim = dimension_picker("cmp_dim")
    t1, t2, t3 = st.tabs(["🔗 Composition AB", "↩️ Inverse A⁻¹", "🔁 Transpose Aᵀ"])
    with t1:
        _composition(dim)
    with t2:
        _inverse(dim)
    with t3:
        _transpose(dim)
