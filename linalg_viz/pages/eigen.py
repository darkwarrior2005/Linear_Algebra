"""Eigenvectors & eigenvalues: the directions a matrix only stretches."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..presets import MATRICES
from ..ui import dimension_picker, fmt, fmt_vec, header, matrix_editor, show, table, tip, vector_editor
from ..viz import plot as P


def _fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def _real_eigvecs(A):
    return [(p.value.real, np.real(p.vector) / np.linalg.norm(np.real(p.vector)))
            for p in la.eigen_analysis(A) if p.is_real]


def _fan_animation(A: np.ndarray, dim: int, span: float, count: int) -> go.Figure:
    eig = _real_eigvecs(A)
    head = span * (0.04 if dim == 2 else 0.06)
    if dim == 2:
        th = np.linspace(0, 2 * np.pi, count, endpoint=False)
        U = np.stack([np.cos(th), np.sin(th)], axis=1)
    else:
        U = _fibonacci_sphere(count)

    def frame(t):
        M = la.interpolate_matrix(A, t)
        tr = []
        if dim == 2:
            tr.append(P.grid_2d(M, 8, color="rgba(77,171,247,0.25)"))
            tr += P.arrows_2d(U @ M.T, color="rgba(220,220,220,0.7)", name="ordinary vectors (get knocked off their line)",
                              width=1.6, head=head * 0.7)
            for k in range(2):
                if k < len(eig):
                    lam, v = eig[k]
                    tr += P.arrows_2d(M @ v, color=P.EIGEN_COLOR if k == 0 else P.color(3),
                                      name=f"eigenvector, λ = {fmt(lam)}", width=5, head=head)
                else:
                    tr += P.arrows_2d(np.zeros((0, 2)), name="")
        else:
            tr += P.arrows_3d(U @ M.T, color="rgba(220,220,220,0.6)", name="ordinary vectors", width=2, head=head * 0.6)
            for k in range(3):
                if k < len(eig):
                    lam, v = eig[k]
                    tr += P.arrows_3d(M @ v, color=[P.EIGEN_COLOR, P.color(3), P.color(4)][k],
                                      name=f"eigenvector, λ = {fmt(lam)}", width=9, head=head)
                else:
                    tr += P.arrows_3d(np.zeros((0, 3)), name="")
        return tr

    static = []
    if dim == 2:
        static.append(P.curve_2d(P.circle_points(), "rgba(255,255,255,0.25)", "", 1, showlegend=False))
        for k, (lam, v) in enumerate(eig):
            static.append(P.line_through_origin_2d(v, span, P.EIGEN_COLOR if k == 0 else P.color(3), ""))
        layout = P.layout_2d(span, 640)
    else:
        static.append(P.axes_3d(span))
        for k, (lam, v) in enumerate(eig):
            static.append(P.line_3d(v, span, [P.EIGEN_COLOR, P.color(3), P.color(4)][k], ""))
        layout = P.layout_3d(span, 640)
    return P.animated_figure(frame, np.linspace(0, 1, 30), layout, static, duration=50 if dim == 2 else 80,
                             redraw=dim == 3)


def _probe(A: np.ndarray, dim: int, span: float) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        if dim == 2:
            theta = st.slider("Direction of v (degrees)", 0.0, 360.0, 30.0, 1.0, key="eig_theta")
            v = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
        else:
            az = st.slider("Azimuth (°)", 0.0, 360.0, 30.0, 1.0, key="eig_az")
            el = st.slider("Elevation (°)", -90.0, 90.0, 20.0, 1.0, key="eig_el")
            a, e = np.radians(az), np.radians(el)
            v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    Av = A @ v
    ang = la.angle_between(v, Av) if np.linalg.norm(Av) > 1e-9 else 0.0
    parallel = np.linalg.norm(np.cross(np.append(v, 0)[:3], np.append(Av, 0)[:3])) < 1e-2 * max(1, np.linalg.norm(Av))
    head = span * (0.045 if dim == 2 else 0.07)
    arrows = P.arrows_2d if dim == 2 else P.arrows_3d
    tr = [] if dim == 2 else [P.axes_3d(span)]
    for k, (lam, ev) in enumerate(_real_eigvecs(A)):
        tr.append(P.line_through_origin_2d(ev, span, P.EIGEN_COLOR, f"eigenline λ = {fmt(lam)}") if dim == 2
                  else P.line_3d(ev, span, P.EIGEN_COLOR, f"eigenline λ = {fmt(lam)}"))
    tr += arrows(v, color=P.color(0), name="v", head=head)
    tr += arrows(Av, color="#ffffff" if not parallel else P.color(5), name="Av", head=head, width=5)
    if dim == 2:
        tr.append(P.line_through_origin_2d(v, span, "rgba(255,212,59,0.25)", "", "dot", 1, False))
    lay = P.layout_2d(span, 520) if dim == 2 else P.layout_3d(span, 520)
    with right:
        show(go.Figure(tr, lay), key="eig_probe_fig")
    with left:
        st.metric("angle between v and Av", f"{fmt(ang, 1)}°")
        if parallel:
            lam = float(np.dot(Av, v))
            st.success(f"Av is on the same line as v: **v is an eigenvector** with λ ≈ {fmt(lam)}")
        else:
            st.caption("Turn v until Av lines up with it (angle 0° or 180°): that direction is an eigenvector.")
    if dim == 2:
        th = np.linspace(0, 360, 721)
        V = np.stack([np.cos(np.radians(th)), np.sin(np.radians(th))])
        W = A @ V
        cross = V[0] * W[1] - V[1] * W[0]
        fig = go.Figure([go.Scatter(x=th, y=cross, mode="lines", line=dict(color="#4dabf7"),
                                    name="sideways part of Av (v × Av)"),
                         go.Scatter(x=[theta], y=[float(v[0] * Av[1] - v[1] * Av[0])], mode="markers",
                                    marker=dict(size=11, color=P.color(0)), name="your v")],
                        P.layout_2d([0, 360, float(cross.min()) - 0.5, float(cross.max()) + 0.5], 260))
        fig.update_yaxes(scaleanchor=None)
        fig.update_layout(xaxis_title="direction of v (°)", showlegend=True)
        show(fig, key="eig_cross_fig")
        st.caption("Where this curve crosses zero, Av has no sideways part: those angles are the eigen-directions.")


def _power(A: np.ndarray, dim: int, span: float) -> None:
    left, right = st.columns([1, 2.3])
    with left:
        v0 = vector_editor("pow_v0", dim, [1, -0.3, 0.5][:dim], "start v₀", P.color(0))
        steps = st.slider("Multiplications", 1, 30, 12, key="pow_steps")
        normalise = st.checkbox("Rescale to length 1 after each step", True, key="pow_norm")
    if v0 is None or np.linalg.norm(v0) < 1e-12:
        return
    seq = [v0 / np.linalg.norm(v0)]
    for _ in range(steps):
        w = A @ seq[-1]
        n = np.linalg.norm(w)
        if n < 1e-14:
            break
        seq.append(w / n if normalise else w)
    seq = np.array(seq)
    eig = _real_eigvecs(A)
    head = span * (0.045 if dim == 2 else 0.07)

    def frame(k):
        k = int(k)
        tr = []
        if dim == 2:
            tr.append(go.Scatter(x=seq[: k + 1, 0], y=seq[: k + 1, 1], mode="markers+lines",
                                 line=dict(color="rgba(255,212,59,0.4)", dash="dot"),
                                 marker=dict(color=np.arange(k + 1), colorscale="Plasma", size=8), name="tips so far"))
            tr += P.arrows_2d(seq[k], color=P.color(0), name=f"A^{k} v₀", head=head, width=4)
        else:
            tr.append(go.Scatter3d(x=seq[: k + 1, 0], y=seq[: k + 1, 1], z=seq[: k + 1, 2], mode="markers+lines",
                                   line=dict(color="rgba(255,212,59,0.5)"), marker=dict(size=4, color=P.color(0)),
                                   name="tips so far"))
            tr += P.arrows_3d(seq[k], color=P.color(0), name=f"A^{k} v₀", head=head)
        return tr

    static = []
    for lam, v in eig:
        static.append(P.line_through_origin_2d(v, span, P.EIGEN_COLOR, f"eigenline λ = {fmt(lam)}") if dim == 2
                      else P.line_3d(v, span, P.EIGEN_COLOR, f"eigenline λ = {fmt(lam)}"))
    if dim == 3:
        static.append(P.axes_3d(span))
    lim = span if normalise else max(span, float(np.nanmax(np.abs(seq))) * 1.1)
    lay = P.layout_2d(lim, 560) if dim == 2 else P.layout_3d(lim, 560)
    fig = P.animated_figure(frame, range(len(seq)), lay, static, duration=350, label=lambda k: str(int(k)),
                            redraw=dim == 3, slider_prefix="step ")
    with right:
        show(fig, key="eig_power_fig")
    w = A @ seq[-1]
    rayleigh = float(np.dot(seq[-1], w) / np.dot(seq[-1], seq[-1]))
    st.markdown(f"After {len(seq) - 1} steps the direction is {fmt_vec(seq[-1] / np.linalg.norm(seq[-1]))}; "
                f"estimated eigenvalue (vᵀAv / vᵀv) ≈ **{fmt(rayleigh)}**.")
    tip("Multiply by A again and again: the part along the eigenvector with the biggest |λ| grows fastest, so "
        "the arrow swings onto that eigenline. This is the *power method* — how Google's PageRank was computed.")


def _char_poly(A: np.ndarray) -> None:
    coeffs = la.characteristic_polynomial(A)
    vals = np.linalg.eigvals(A)
    st.latex(r"\det(\lambda I - A) = " + la.polynomial_to_text(coeffs).replace("λ", r"\lambda "))
    real = vals[np.abs(vals.imag) < 1e-9].real
    lo = min(-3.0, float(np.min(vals.real)) - 1.5)
    hi = max(3.0, float(np.max(vals.real)) + 1.5)
    lam = np.linspace(lo, hi, 600)
    p = np.polyval(np.real(coeffs), lam)
    clip = np.percentile(np.abs(p), 90) * 1.5 + 1
    a, b = st.columns(2)
    with a:
        fig = go.Figure([go.Scatter(x=lam, y=np.clip(p, -clip, clip), mode="lines", line=dict(color="#4dabf7", width=3),
                                    name="p(λ)"),
                         go.Scatter(x=real, y=np.zeros_like(real), mode="markers", marker=dict(size=13, color=P.EIGEN_COLOR),
                                    name="roots = eigenvalues")],
                        P.layout_2d([lo, hi, -clip, clip], 380, "Characteristic polynomial"))
        fig.update_yaxes(scaleanchor=None)
        show(fig, key="eig_poly")
    with b:
        r = max(1.5, float(np.max(np.abs(vals))) * 1.3)
        fig = go.Figure([P.curve_2d(P.circle_points(200, 1), "rgba(255,255,255,0.3)", "|λ| = 1", 1),
                         go.Scatter(x=vals.real, y=vals.imag, mode="markers+text", text=[fmt(v) for v in vals],
                                    textposition="top center", marker=dict(size=13, color=P.EIGEN_COLOR),
                                    name="eigenvalues")],
                        P.layout_2d(r, 380, "Eigenvalues in the complex plane"))
        fig.update_layout(xaxis_title="real part", yaxis_title="imaginary part")
        show(fig, key="eig_cplx")


def _table(A: np.ndarray) -> None:
    pairs = la.eigen_analysis(A)
    rows = []
    for p in pairs:
        lam = p.value
        if p.is_real:
            lr = lam.real
            meaning = ("stays put" if abs(lr - 1) < 1e-9 else "is crushed to 0" if abs(lr) < 1e-9 else
                       f"is stretched ×{fmt(lr)}" if lr > 1 else f"is shrunk ×{fmt(lr)}" if lr > 0 else
                       f"is flipped and scaled ×{fmt(abs(lr))}")
            vec = fmt_vec(np.real(p.vector))
        else:
            meaning = (f"rotation by {fmt(np.degrees(abs(np.angle(lam))), 1)}° combined with scaling ×{fmt(abs(lam))}"
                       " — no real direction survives")
            vec = "complex: " + fmt_vec(p.vector)
        rows.append({"λ": fmt(lam), "eigenvector": vec, "along it, space…": meaning,
                     "alg. mult.": p.algebraic_multiplicity, "geo. mult.": p.geometric_multiplicity})
    table(rows)
    diag = la.is_diagonalizable(A)
    st.markdown(f"Sum of eigenvalues = trace = **{fmt(np.trace(A))}**, product = determinant = "
                f"**{fmt(np.linalg.det(A))}**. "
                + ("Diagonalizable ✅ (enough eigenvectors to form a basis)." if diag else
                   "Not diagonalizable ⚠️ — a repeated eigenvalue without enough eigenvectors (a shear-like map)."))


def render() -> None:
    header("Eigenvectors & eigenvalues",
           "Most vectors get knocked off their line by a matrix. Eigenvectors stay on it — only stretched by λ.")
    dim = dimension_picker("eig_dim")
    left, right = st.columns([1, 2.3])
    with left:
        A = matrix_editor("eig_A", dim, dim, MATRICES[dim]["Stretch along eigenvectors" if dim == 2 else "Symmetric"],
                          MATRICES[dim], label="A")
        span = st.slider("View size", 1.5, 10.0, 3.5 if dim == 2 else 2.5, 0.5, key=f"eig_span{dim}")
        count = st.slider("Number of test vectors", 8, 60, 24 if dim == 2 else 30, key=f"eig_count{dim}")
    if A is None:
        return
    with right:
        show(_fan_animation(A, dim, span, count), key="eig_fan")
    if not _real_eigvecs(A):
        st.warning("No real eigenvectors: every direction gets rotated. The eigenvalues are complex — see the table.")
    _table(A)
    tip("Press ▶ Play. Grey arrows turn as A is applied; the coloured eigenvectors keep pointing along their "
        "dashed lines and only grow, shrink or flip. The stretch factor is the eigenvalue λ:  Av = λv.")
    t1, t2, t3 = st.tabs(["🔦 Hunt for eigenvectors", "🔁 Multiply again and again", "📉 Characteristic polynomial"])
    with t1:
        _probe(A, dim, span)
    with t2:
        _power(A, dim, span)
    with t3:
        _char_poly(A)
