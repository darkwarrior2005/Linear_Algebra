"""Orthogonality: Gram–Schmidt step by step and projections onto subspaces."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..ui import dimension_picker, fmt, fmt_vec, header, latex_matrix, matrix_editor, show, tip, vector_editor
from ..viz import plot as P


def _gram_schmidt(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        k = st.radio("How many vectors", list(range(2, dim + 1)), index=dim - 2, horizontal=True, key=f"gs_k{dim}")
        default = [[2, 1, 1], [1, 2, 0], [0, 1, 2]] if dim == 3 else [[3, 1], [1, 2]]
        V = matrix_editor("gs_V", dim, k, [row[:k] for row in default[:dim]], label="Vectors as columns")
        span = st.slider("View size", 1.5, 8.0, 3.0, 0.5, key=f"gs_span{dim}")
    if V is None:
        return
    steps = la.gram_schmidt(V)
    with left:
        stage = st.slider("Step", 1, len(steps), len(steps), key=f"gs_stage{dim}{k}",
                          help="Step j makes v_j perpendicular to everything before it.")
    arrows = P.arrows_2d if dim == 2 else P.arrows_3d
    head = span * (0.045 if dim == 2 else 0.07)
    tr: list = [] if dim == 2 else [P.axes_3d(span)]
    for s in steps:
        tr += arrows(s.original, color="rgba(200,200,200,0.5)", name=f"v{s.index + 1}" if s.index == 0 else "",
                     head=head, labels=[f"v{s.index + 1}"], showlegend=s.index == 0)
    for s in steps[: stage - 1]:
        if s.normalised is not None:
            tr += arrows(s.normalised, color=P.BASIS_COLORS[s.index], name=f"q{s.index + 1}", head=head,
                         labels=[f"q{s.index + 1}"])
    cur = steps[stage - 1]
    tr += arrows(cur.original, color=P.color(0), name=f"v{cur.index + 1} (current)", head=head)
    base = np.zeros(dim)
    for j, p in enumerate(cur.projections):
        tr += arrows(-p, cur.original - base, color=P.color(1), name=f"− projection onto q{j + 1}" if j == 0 else "",
                     head=head, showlegend=j == 0)
        base = base + p
    tr += arrows(cur.orthogonal, color="#ffffff", name="what is left (perpendicular part)", head=head, width=5)
    if cur.normalised is not None:
        tr += arrows(cur.normalised, color=P.BASIS_COLORS[cur.index], name=f"q{cur.index + 1} = unit length",
                     head=head, labels=[f"q{cur.index + 1}"])
    if dim == 3 and cur.index >= 1:
        q = [s.normalised for s in steps[: cur.index] if s.normalised is not None]
        if len(q) == 2:
            tr.append(P.plane_through_origin(q[0], q[1], span, "#4dabf7", "span of q₁, q₂", 0.15))
        elif len(q) == 1:
            tr.append(P.line_3d(q[0], span, "#4dabf7", "span of q₁"))
    lay = P.layout_2d(span, 600) if dim == 2 else P.layout_3d(span, 600)
    with right:
        show(go.Figure(tr, lay), key="gs_fig")
    if cur.normalised is None:
        st.warning(f"v{cur.index + 1} is a combination of the earlier vectors — nothing is left after removing "
                   "the projections, so it adds no new direction.")
    st.latex(rf"w_{{{cur.index + 1}}} = v_{{{cur.index + 1}}} - "
             + (" - ".join(rf"(v_{{{cur.index + 1}}}\cdot q_{{{j + 1}}})q_{{{j + 1}}}"
                           for j in range(len(cur.projections))) or "0")
             + rf",\quad q_{{{cur.index + 1}}} = \frac{{w_{{{cur.index + 1}}}}}{{|w_{{{cur.index + 1}}}|}}")
    if la.rank(V) == k:
        Q, R = la.qr_decomposition(V)
        c1, c2 = st.columns(2)
        with c1:
            latex_matrix(Q, "Q = ")
        with c2:
            latex_matrix(R, "R = ")
        st.caption("Gram–Schmidt is the QR decomposition: V = QR, with orthonormal columns in Q and the "
                   "recipe for rebuilding each vᵢ in the upper-triangular R.")
    tip("Each step removes the shadow of the new vector on all the previous directions; what is left points "
        "in a brand-new perpendicular direction. Shrinking it to length 1 gives the next q.")


def _projection(dim: int) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        v = vector_editor("prj_v", dim, [1, 2, 3][:dim], "v", P.color(0))
        k = 1 if dim == 2 else st.radio("Project onto", [1, 2], format_func=lambda x: "a line" if x == 1 else "a plane",
                                        index=1, horizontal=True, key="prj_k")
        default = [[2, 1], [1, 0], [0, 1]] if dim == 3 else [[2], [1]]
        S = matrix_editor("prj_S", dim, k, [row[:k] for row in default[:dim]], label="Subspace spanned by columns")
        span = st.slider("View size", 1.5, 8.0, 4.0, 0.5, key=f"prj_span{dim}")
    if v is None or S is None or la.rank(S) == 0:
        return
    Pm = la.projection_matrix(S)
    p = Pm @ v
    e = v - p
    arrows = P.arrows_2d if dim == 2 else P.arrows_3d
    head = span * (0.045 if dim == 2 else 0.07)
    Q = la.column_space(S)
    if dim == 2:
        tr = [P.line_through_origin_2d(Q[:, 0], span, "#4dabf7", "subspace", "solid", 3)]
    else:
        tr = [P.axes_3d(span)]
        tr.append(P.plane_through_origin(Q[:, 0], Q[:, 1], span, "#4dabf7", "subspace", 0.3) if Q.shape[1] == 2
                  else P.line_3d(Q[:, 0], span, "#4dabf7", "subspace", 6, None))
    tr += arrows(v, color=P.color(0), name=f"v = {fmt_vec(v)}", head=head)
    tr += arrows(p, color="#ffffff", name=f"projection Pv = {fmt_vec(p)}", head=head, width=5)
    tr += arrows(e, p, color=P.color(4), name=f"error v − Pv (⊥ subspace), |e| = {fmt(np.linalg.norm(e))}", head=head)
    lay = P.layout_2d(span, 600) if dim == 2 else P.layout_3d(span, 600)
    with right:
        show(go.Figure(tr, lay), key="prj_fig")
    latex_matrix(Pm, r"P = A(A^TA)^{-1}A^T = ")
    st.markdown(f"P² = P: {'✅' if np.allclose(Pm @ Pm, Pm) else '❌'}  ·  Pᵀ = P: "
                f"{'✅' if np.allclose(Pm, Pm.T) else '❌'}  ·  error · subspace = "
                f"{fmt(float(np.max(np.abs(S.T @ e))))}")
    tip("The projection is the closest point of the subspace to v; the error arrow is perpendicular to the "
        "subspace. This is exactly how least-squares fitting works.")


def render() -> None:
    header("Gram–Schmidt & projections", "Turn any basis into perpendicular unit vectors; drop shadows onto subspaces.")
    dim = dimension_picker("orth_dim", default=3)
    t1, t2 = st.tabs(["📏 Gram–Schmidt", "🔦 Projection onto a subspace"])
    with t1:
        _gram_schmidt(dim)
    with t2:
        _projection(dim)
