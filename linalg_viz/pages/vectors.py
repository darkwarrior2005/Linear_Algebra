"""Vectors: what they are, adding, scaling, combining, span, dot & cross products."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..ui import dimension_picker, fmt, fmt_vec, header, show, tip, vector_editor
from ..viz import plot as P


def _arrows(dim, vectors, origins=None, **kw):
    if dim == 2:
        return P.arrows_2d(vectors, origins, **kw)
    kw.pop("dash", None)
    return P.arrows_3d(vectors, origins, **kw)


def _layout(dim, span, height=620):
    return P.layout_2d(span, height) if dim == 2 else P.layout_3d(span, height)


def _segment(dim, a, b, color, name="", dash="dot"):
    if dim == 2:
        return go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color=color, dash=dash, width=1.5),
                          name=name, showlegend=bool(name), hoverinfo="skip")
    return go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                        line=dict(color=color, dash=dash, width=3), name=name, showlegend=bool(name),
                        hoverinfo="skip")


def _add_and_scale(dim: int) -> None:
    left, right = st.columns([1, 2.2])
    with left:
        u = vector_editor("vec_u", dim, [2, 1, 0][:dim], "u", P.color(0))
        v = vector_editor("vec_v", dim, [-1, 2, 1][:dim], "v", P.color(1))
        a = st.slider("a (scales u)", -3.0, 3.0, 1.0, 0.1, key="vec_a")
        b = st.slider("b (scales v)", -3.0, 3.0, 1.0, 0.1, key="vec_b")
        show_diff = st.checkbox("Also show u − v", key="vec_diff")
        span = st.slider("View size", 2.0, 12.0, 5.0, 0.5, key=f"vec_span{dim}")
    if u is None or v is None:
        return
    au, bv = a * u, b * v
    w = au + bv
    head = span * 0.05
    traces = []
    if dim == 3:
        traces.append(P.axes_3d(span))
    traces += _arrows(dim, au, color=P.color(0), name=f"a·u = {fmt_vec(au)}", head=head)
    traces += _arrows(dim, bv, color=P.color(1), name=f"b·v = {fmt_vec(bv)}", head=head)
    # tip-to-tail: b·v starting at the tip of a·u
    traces += _arrows(dim, bv, au, color=P.color(1), name="", head=head, opacity=0.45, showlegend=False)
    traces += _arrows(dim, au, bv, color=P.color(0), name="", head=head, opacity=0.45, showlegend=False)
    traces += _arrows(dim, w, color="#ffffff", name=f"a·u + b·v = {fmt_vec(w)}", head=head)
    if show_diff:
        traces += _arrows(dim, u - v, v, color=P.color(4), name=f"u − v = {fmt_vec(u - v)} (from v to u)", head=head)
    fig = go.Figure(traces, _layout(dim, span))
    with right:
        show(fig, key="vec_add_fig")
    tip("A vector is an arrow from the origin: its entries say how far to go along each axis. "
        "Adding vectors = walking one arrow, then the other (tip to tail). Scaling stretches, "
        "shrinks or flips an arrow along its own line.")
    c1, c2, c3 = st.columns(3)
    c1.metric("|u|", fmt(np.linalg.norm(u)))
    c2.metric("|v|", fmt(np.linalg.norm(v)))
    c3.metric("|a·u + b·v|", fmt(np.linalg.norm(w)))


def _span(dim: int) -> None:
    left, right = st.columns([1, 2.2])
    with left:
        count = st.radio("Number of vectors", [1, 2, 3] if dim == 3 else [1, 2, 3], index=1, horizontal=True,
                         key=f"span_count{dim}")
        defaults = [[1, 0.5, 0], [0.5, 1, 0.5], [1, 1.5, 0.5]]
        vecs = []
        for k in range(count):
            vecs.append(vector_editor(f"span_v{k}", dim, defaults[k][:dim], f"v{k + 1}", P.color(k)))
        coeffs = [st.slider(f"c{k + 1}", -3.0, 3.0, [1.5, -1.0, 0.5][k], 0.1, key=f"span_c{k}") for k in range(count)]
        show_lattice = st.checkbox("Show many combinations (dots)", True, key="span_lattice")
        span = st.slider("View size", 2.0, 12.0, 5.0, 0.5, key=f"span_span{dim}")
    if any(v is None for v in vecs):
        return
    V = np.column_stack(vecs)
    r = la.rank(V)
    head = span * 0.05
    traces = []
    if dim == 3:
        traces.append(P.axes_3d(span))
    # The span itself
    basis = la.column_space(V)
    if r == 1:
        traces.append(P.line_through_origin_2d(basis[:, 0], span, "rgba(255,212,59,0.7)", "span (a line)")
                      if dim == 2 else P.line_3d(basis[:, 0], span, "rgba(255,212,59,0.8)", "span (a line)"))
    elif r == 2 and dim == 3:
        traces.append(P.plane_through_origin(basis[:, 0], basis[:, 1], span, "#ffd43b", "span (a plane)", 0.25))
    if show_lattice and r > 0:
        g = np.linspace(-3, 3, 13 if count < 3 else 7)
        mesh = np.meshgrid(*([g] * count), indexing="ij")
        C = np.stack([m.ravel() for m in mesh])
        pts = (V @ C).T
        if dim == 2:
            traces.append(P.points_2d(pts, "rgba(255,255,255,0.35)", "c₁v₁ + c₂v₂ + … for many c", 4))
        else:
            traces.append(P.points_3d(pts, "rgba(255,255,255,0.35)", "combinations", 2))
    combo = V @ np.array(coeffs)
    origin = np.zeros(dim)
    for k, v in enumerate(vecs):
        traces += _arrows(dim, v, color=P.color(k), name=f"v{k + 1}", head=head)
        traces += _arrows(dim, coeffs[k] * v, origin, color=P.color(k), name="", head=head, opacity=0.5,
                          showlegend=False)
        origin = origin + coeffs[k] * v
    traces += _arrows(dim, combo, color="#ffffff", name=f"combination = {fmt_vec(combo)}", head=head)
    fig = go.Figure(traces, _layout(dim, span))
    with right:
        show(fig, key="span_fig")
    independent = r == count
    what = {0: "just the origin", 1: "a line", 2: "a plane" if dim == 3 else "the whole plane", 3: "all of 3D space"}[r]
    st.markdown(f"**Rank {r}** — these vectors span **{what}**. They are "
                f"**{'linearly independent' if independent else 'linearly dependent'}**"
                + ("." if independent else " (at least one is a combination of the others)."))
    tip("The span is every place you can reach with c₁v₁ + c₂v₂ + …  Drag the sliders: the white arrow "
        "is always stuck inside the span. A dependent vector adds no new direction.")


def _products(dim: int) -> None:
    left, right = st.columns([1, 2.2])
    with left:
        u = vector_editor("dot_u", dim, [3, 1, 0.5][:dim], "u", P.color(0))
        v = vector_editor("dot_v", dim, [1, 2, 1][:dim], "v", P.color(1))
        span = st.slider("View size", 2.0, 12.0, 5.0, 0.5, key=f"dot_span{dim}")
    if u is None or v is None:
        return
    head = span * 0.05
    proj = la.project_onto_vector(u, v)
    traces = []
    if dim == 3:
        traces.append(P.axes_3d(span))
    traces += _arrows(dim, u, color=P.color(0), name="u", head=head)
    traces += _arrows(dim, v, color=P.color(1), name="v", head=head)
    traces += _arrows(dim, proj, color="#ffffff", name="projection of u onto v", head=head, width=5)
    traces.append(_segment(dim, u, proj, "#aaaaaa", "u − proj (⊥ to v)"))
    if dim == 2:
        traces.append(P.polygon_2d(np.array([[0, 0], u, u + v, v]), "#cc5de8", "parallelogram (area = |det|)", 0.15))
    else:
        cross = np.cross(u, v)
        traces += P.arrows_3d(cross, color=P.color(3), name=f"u × v = {fmt_vec(cross)}", head=head)
        quad = np.array([[0, 0, 0], u, u + v, v])
        traces.append(go.Mesh3d(x=quad[:, 0], y=quad[:, 1], z=quad[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                                color="#cc5de8", opacity=0.25, name="parallelogram", showlegend=True,
                                hoverinfo="skip"))
    fig = go.Figure(traces, _layout(dim, span))
    with right:
        show(fig, key="dot_fig")
    dot = float(np.dot(u, v))
    c = st.columns(4)
    c[0].metric("u · v", fmt(dot))
    c[1].metric("angle", f"{fmt(la.angle_between(u, v), 1)}°")
    c[2].metric("|proj|", fmt(np.linalg.norm(proj)))
    if dim == 2:
        c[3].metric("signed area u×v", fmt(u[0] * v[1] - u[1] * v[0]))
    else:
        c[3].metric("|u × v| = area", fmt(np.linalg.norm(np.cross(u, v))))
    tip("Dot product = (length of the shadow of u on v) × |v|. It is 0 exactly when the vectors are "
        "perpendicular and negative when they point apart. The cross product (3D) is perpendicular to "
        "both, and its length is the parallelogram's area.")


def render() -> None:
    header("Vectors", "Arrows you can add, scale, combine and multiply — in 2D or 3D.")
    dim = dimension_picker("vectors_dim")
    t1, t2, t3 = st.tabs(["➕ Add & scale", "🕸️ Linear combinations & span", "📐 Dot, cross & projection"])
    with t1:
        _add_and_scale(dim)
    with t2:
        _span(dim)
    with t3:
        _products(dim)
