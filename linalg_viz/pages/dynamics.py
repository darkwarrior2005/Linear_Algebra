"""Linear dynamical systems x' = Ax: eigenvalues decide how everything moves."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..presets import LINEAR_SYSTEMS
from ..ui import dimension_picker, fmt, header, latex_matrix, matrix_editor, show, table, tip
from ..viz import plot as P
from ..viz.scenes import SceneOptions, transform_figure


def _trajectories(A: np.ndarray, starts: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """x(t) = e^{tA} x0 for each start, shape (len(ts), n_starts, dim)."""
    dt = ts[1] - ts[0]
    step = la.matrix_exp(A * dt)
    out = np.empty((len(ts),) + starts.shape)
    x = starts @ la.matrix_exp(A * ts[0]).T
    for k in range(len(ts)):
        out[k] = x
        x = x @ step.T
    return out


def _trace_det_plane(A: np.ndarray) -> go.Figure:
    tr, det = float(np.trace(A)), float(np.linalg.det(A))
    T = np.linspace(-4, 4, 200)
    fig = go.Figure([
        go.Scatter(x=T, y=T ** 2 / 4, mode="lines", line=dict(color="#adb5bd", dash="dash"), name="det = tr²/4"),
        go.Scatter(x=[tr], y=[det], mode="markers", marker=dict(size=14, color=P.EIGEN_COLOR), name="your matrix"),
    ], P.layout_2d([-4, 4, -3, 5], 380, "Trace–determinant map"))
    fig.update_yaxes(scaleanchor=None, title="det A")
    fig.update_xaxes(title="trace A")
    for x, y, text in [(0, -2, "saddles"), (-2.6, 0.6, "stable nodes"), (2.6, 0.6, "unstable nodes"),
                       (-1.2, 3.5, "stable spirals"), (1.2, 3.5, "unstable spirals"), (0, 4.6, "centres on tr = 0")]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color="#868e96"))
    return fig


def render() -> None:
    header("Linear dynamics  x′ = Ax",
           "A matrix can also be a velocity rule. Its eigenvalues decide whether things spiral, explode or settle.")
    dim = dimension_picker("dyn_dim")
    left, right = st.columns([1, 2.3])
    with left:
        default = LINEAR_SYSTEMS[dim]["Stable spiral" if dim == 2 else "Saddle-spiral"]
        A = matrix_editor("dyn_A", dim, dim, default, LINEAR_SYSTEMS[dim], label="A")
        T = st.slider("Time to run", 1.0, 20.0, 8.0, 0.5, key=f"dyn_T{dim}")
        n_traj = st.slider("Number of trajectories", 4, 60, 16 if dim == 2 else 20, key=f"dyn_n{dim}")
        span = st.slider("View size", 1.0, 10.0, 3.0, 0.5, key=f"dyn_span{dim}")
    if A is None:
        return
    rng = np.random.default_rng(3)
    if dim == 2:
        th = np.linspace(0, 2 * np.pi, n_traj, endpoint=False)
        starts = span * 0.9 * np.stack([np.cos(th), np.sin(th)], 1)
    else:
        v = rng.normal(size=(n_traj, 3))
        starts = span * 0.8 * v / np.linalg.norm(v, axis=1, keepdims=True)
    ts = np.linspace(0, T, 241)
    paths = _trajectories(A, starts, ts)
    paths[np.abs(paths) > span * 50] = np.nan
    eig = [(p.value.real, np.real(p.vector)) for p in la.eigen_analysis(A) if p.is_real]
    idx = np.linspace(0, len(ts) - 1, 61).astype(int)

    def frame(k):
        k = int(k)
        Pk = paths[idx[k]]
        if dim == 2:
            return [go.Scatter(x=Pk[:, 0], y=Pk[:, 1], mode="markers", marker=dict(size=9, color=P.color(0)),
                               name="state x(t)")]
        return [go.Scatter3d(x=Pk[:, 0], y=Pk[:, 1], z=Pk[:, 2], mode="markers", marker=dict(size=4, color=P.color(0)),
                             name="state x(t)")]

    static = []
    if dim == 2:
        g = np.linspace(-span, span, 19)
        X, Y = np.meshgrid(g, g)
        UV = np.einsum("ij,jkl->ikl", A, np.stack([X, Y]))
        static += P.quiver_2d(X, Y, UV[0], UV[1], 2 * span / 19 * 0.8, colorscale="Greys")[:-1]
        S = P.nan_join([paths[:, i] for i in range(n_traj)])
        static.append(go.Scatter(x=S[:, 0], y=S[:, 1], mode="lines", line=dict(color="rgba(77,171,247,0.7)", width=1.6),
                                 name="trajectories", hoverinfo="skip"))
        for lam, v in eig:
            static.append(P.line_through_origin_2d(v, span, P.EIGEN_COLOR, f"eigenline λ = {fmt(lam)} "
                                                   f"({'grows' if lam > 0 else 'decays' if lam < 0 else 'still'})"))
        lay = P.layout_2d(span, 640)
    else:
        S = P.nan_join([paths[:, i] for i in range(n_traj)], 3)
        static.append(go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="lines", line=dict(color="rgba(77,171,247,0.7)", width=3),
                                   name="trajectories", hoverinfo="skip"))
        static.append(P.axes_3d(span))
        for lam, v in eig:
            static.append(P.line_3d(v, span, P.EIGEN_COLOR, f"eigenline λ = {fmt(lam)}"))
        lay = P.layout_3d(span, 640)
    fig = P.animated_figure(frame, range(len(idx)), lay, static, duration=60, initial=0,
                            label=lambda k: f"{ts[idx[int(k)]]:.2f}", redraw=dim == 3)
    with right:
        show(fig, key="dyn_fig")

    vals = np.linalg.eigvals(A)
    rows = []
    for v in vals:
        growth = ("grows like e^{" + fmt(v.real) + "t}" if v.real > 1e-9 else
                  "decays like e^{" + fmt(v.real) + "t}" if v.real < -1e-9 else "neither grows nor decays")
        osc = f", oscillates with period {fmt(2 * np.pi / abs(v.imag))}" if abs(v.imag) > 1e-9 else ""
        rows.append({"eigenvalue": fmt(v), "behaviour along it": growth + osc})
    a, b = st.columns([1.3, 1])
    with a:
        if dim == 2:
            st.markdown(f"### {la.classify_linear_system_2d(A)}")
        else:
            st.markdown("### " + ("Stable: everything returns to 0" if np.all(vals.real < -1e-9) else
                                  "Unstable: some motions blow up" if np.any(vals.real > 1e-9) else
                                  "Neutral: some motions neither grow nor decay"))
        table(rows)
        st.latex(r"x(t) = e^{tA}x_0")
        latex_matrix(la.matrix_exp(A), "e^{A} = ")
    with b:
        if dim == 2:
            show(_trace_det_plane(A), key="dyn_td")
    st.markdown("#### The whole space flowing:  x ↦ e^{tA} x")
    o = SceneOptions(span=span, circle=dim == 2, height=520)
    show(transform_figure(dim, lambda t: la.matrix_exp(A * t), np.linspace(0, min(T, 4.0), 41), o, initial=0,
                          slider_prefix="t = "), key="dyn_space")
    tip("Along an eigenvector the motion is simply x(t) = e^{λt}x₀ — straight in or out. Complex eigenvalues "
        "a ± bi make things rotate (frequency b) while growing or shrinking (rate a). That is why the "
        "eigenvalues alone tell you whether a bridge, a circuit or an ecosystem is stable.")
