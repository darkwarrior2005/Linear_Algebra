"""Flows: release particles into a (possibly time-dependent) field and watch them evolve."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core.fields import field_rhs, find_fixed_points, rk4_paths
from ..core.parsing import ExpressionError, parse_matrix
from ..field_ui import Window, field_input
from ..ui import fmt, fmt_vec, header, show, table, tip
from ..viz import plot as P

SEEDINGS = ["grid", "random", "blob (a disc / ball of fluid)", "line", "my own points"]


def _seeds(kind: str, w: Window, count: int, centre: np.ndarray, radius: float, text: str) -> np.ndarray:
    dim = w.dim
    rng = np.random.default_rng(7)
    lo = np.array(w.bounds[0::2])
    hi = np.array(w.bounds[1::2])
    if kind == "grid":
        k = max(2, int(round(count ** (1 / dim))))
        axes = [np.linspace(lo[i] + (hi[i] - lo[i]) * 0.08, hi[i] - (hi[i] - lo[i]) * 0.08, k) for i in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)
    if kind == "random":
        return rng.uniform(lo, hi, (count, dim))
    if kind.startswith("blob"):
        if dim == 2:
            # outline (ordered, for the polygon) followed by interior points
            th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            ring = centre + radius * np.stack([np.cos(th), np.sin(th)], 1)
            r = radius * np.sqrt(rng.uniform(0, 1, count))
            a = rng.uniform(0, 2 * np.pi, count)
            inside = centre + np.stack([r * np.cos(a), r * np.sin(a)], 1)
            return np.vstack([ring, inside])
        v = rng.normal(size=(count, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return centre + radius * v * rng.uniform(0, 1, (count, 1)) ** (1 / 3)
    if kind == "line":
        s = np.linspace(-1, 1, count)[:, None]
        d = np.zeros(dim)
        d[0] = radius
        return centre + s * d
    try:
        pts = parse_matrix(text)
    except (ExpressionError, ValueError) as exc:
        st.error(f"Points: {exc}")
        return np.zeros((0, dim))
    if pts.shape[1] != dim:
        st.error(f"Each point needs {dim} numbers.")
        return np.zeros((0, dim))
    return pts


def _shoelace(P2: np.ndarray) -> float:
    x, y = P2[:, 0], P2[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def render() -> None:
    header("Flows & particles", "Drop particles into a field and let time run: dx/dt = F(x, t).")
    dim = 2 if st.radio("Space", ["2D", "3D"], horizontal=True, key="flow_dim") == "2D" else 3
    left, right = st.columns([1, 2.3])
    with left:
        F, w, name = field_input("flow", dim, "Van der Pol oscillator" if dim == 2 else "Lorenz system")
        if F is None:
            return
        with st.expander("Particles", expanded=True):
            kind = st.selectbox("Starting positions", SEEDINGS, index=2 if dim == 2 else 1, key="flow_seed")
            count = st.slider("How many", 4, 600, 120 if dim == 2 else 60, key=f"flow_count{dim}")
            lo = np.array(w.bounds[0::2])
            hi = np.array(w.bounds[1::2])
            text = ""
            centre, radius = (lo + hi) / 2, float(np.min(hi - lo)) / 8
            if kind.startswith("blob") or kind == "line":
                cols = st.columns(dim)
                centre = np.array([cols[k].number_input(f"centre {'xyz'[k]}", value=float((lo[k] + hi[k]) / 2 +
                                   (hi[k] - lo[k]) * (0.2 if k == 0 else 0.0)), step=0.1, key=f"flow_c{k}{dim}")
                                   for k in range(dim)])
                radius = st.number_input("radius / half-length", 0.01, 100.0, radius, 0.05, key=f"flow_r{dim}")
            if kind == "my own points":
                text = st.text_area("One point per line", "1 0\n0 1\n-1 -1" if dim == 2 else "1 1 1\n-1 2 20",
                                    key=f"flow_pts{dim}")
        with st.expander("Time", expanded=True):
            t0 = st.number_input("start time", value=0.0, step=0.5, key="flow_t0")
            T = st.slider("duration", 0.5, 60.0, 10.0 if dim == 2 else 15.0, 0.5, key=f"flow_T{dim}")
            steps = st.slider("integration steps", 100, 3000, 600 if dim == 2 else 1500, 50, key=f"flow_steps{dim}")
            frames = st.slider("animation frames", 10, 120, 60, key="flow_frames")
            trail = st.slider("trail length (frames)", 0, 120, 12 if dim == 2 else 40, key="flow_trail")
        with st.expander("Extras"):
            show_field = st.checkbox("Show the field arrows", dim == 2, key="flow_field")
            show_fp = st.checkbox("Find fixed points (where F = 0)", not F.time_dependent, key="flow_fp",
                                  disabled=F.time_dependent, help="Only meaningful when F does not change with time.")
    seeds = _seeds(kind, w, count, centre, radius, text)
    if len(seeds) == 0:
        return
    bound = 50 * max(1.0, w.size)
    times, paths = rk4_paths(field_rhs(F), seeds, t0, t0 + T, steps, bound=bound)
    idx = np.unique(np.linspace(0, steps, frames).astype(int))
    blob2d = dim == 2 and kind.startswith("blob")
    n_ring = 120 if blob2d else 0
    colors = np.linspace(0, 1, len(seeds))
    n_q = 16

    def frame(k):
        k = int(k)
        tr = []
        Pk = paths[idx[k]]
        first = idx[max(0, k - trail)]
        seg = paths[first: idx[k] + 1: max(1, (idx[k] - first) // 30 or 1)]
        if dim == 2:
            if show_field:
                (_, _), (X, Y) = w.mesh(n_q)
                U, V = F(X, Y, t=times[idx[k]])
                tr += P.quiver_2d(X, Y, U, V, w.size / n_q * 0.8, colorscale="Greys")[:-1]
            if blob2d:
                ring = Pk[:n_ring]
                tr.append(go.Scatter(x=np.append(ring[:, 0], ring[0, 0]), y=np.append(ring[:, 1], ring[0, 1]),
                                     fill="toself", fillcolor="rgba(255,212,59,0.25)", mode="lines",
                                     line=dict(color="#ffd43b", width=2), name="blob of fluid"))
            S = P.nan_join([seg[:, i] for i in range(n_ring, len(seeds))] if trail and len(seg) > 1 else [])
            tr.append(go.Scatter(x=S[:, 0], y=S[:, 1], mode="lines",
                                 line=dict(color="rgba(77,171,247,0.45)", width=1.2), name="trails", hoverinfo="skip"))
            tr.append(go.Scatter(x=Pk[n_ring:, 0], y=Pk[n_ring:, 1], mode="markers", name="particles",
                                 marker=dict(size=6, color=colors[n_ring:], colorscale="Plasma",
                                             line=dict(width=0))))
        else:
            S = P.nan_join([seg[:, i] for i in range(len(seeds))] if trail and len(seg) > 1 else [], 3)
            tr.append(go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="lines",
                                   line=dict(color="rgba(77,171,247,0.6)", width=2), name="trails", hoverinfo="skip"))
            tr.append(go.Scatter3d(x=Pk[:, 0], y=Pk[:, 1], z=Pk[:, 2], mode="markers", name="particles",
                                   marker=dict(size=3.5, color=colors, colorscale="Plasma")))
        return tr

    static = []
    fps = []
    if show_fp and not F.time_dependent:
        fps = find_fixed_points(F, w.bounds, seeds_per_axis=9 if dim == 2 else 6)
        if fps:
            pts = np.array([f.point for f in fps])
            static.append(P.points_2d(pts, "#ff6b6b", "fixed points", 12, symbol="x") if dim == 2
                          else P.points_3d(pts, "#ff6b6b", "fixed points", 7))
    if dim == 2:
        lay = P.layout_2d(w.layout_range(), 700)
    else:
        lay = P.layout_3d(1, 720)
        finite = paths[np.isfinite(paths).all(axis=2)]
        lo3 = np.minimum(np.array(w.bounds[0::2]), np.nanpercentile(finite, 1, axis=0) if len(finite) else 0)
        hi3 = np.maximum(np.array(w.bounds[1::2]), np.nanpercentile(finite, 99, axis=0) if len(finite) else 0)
        lay.scene.xaxis.range, lay.scene.yaxis.range, lay.scene.zaxis.range = ([lo3[0], hi3[0]], [lo3[1], hi3[1]],
                                                                               [lo3[2], hi3[2]])
        lay.scene.aspectmode = "cube"
    fig = P.animated_figure(frame, range(len(idx)), lay, static, duration=50, label=lambda k: f"{times[idx[int(k)]]:.2f}",
                            initial=0, redraw=dim == 3 or show_field)
    with right:
        show(fig, key="flow_fig")
    lost = int(np.sum(~np.isfinite(paths[-1]).all(axis=1)))
    if lost:
        st.caption(f"{lost} particle(s) flew far away and were removed.")
    if blob2d:
        areas = np.array([_shoelace(paths[i, :n_ring]) for i in idx])
        div_now = F.divergence_at(centre, t0)
        fig2 = go.Figure(go.Scatter(x=times[idx], y=areas, mode="lines", line=dict(color="#ffd43b", width=3),
                                    name="area"), P.layout_2d([times[0], times[-1], 0, float(np.nanmax(areas)) * 1.15 + 1e-9],
                                                              260, "Area of the blob over time"))
        fig2.update_yaxes(scaleanchor=None)
        fig2.update_layout(xaxis_title="t", yaxis_title="area")
        show(fig2, key="flow_area")
        st.caption(f"Start area {fmt(areas[0])} → end area {fmt(areas[-1])}. Divergence at the blob's start: "
                   f"{fmt(div_now)}. Divergence-free flows (like rotations, the pendulum, the double gyre) keep the "
                   "area constant while stretching the shape — that is how fluids mix.")
    if fps:
        st.markdown("**Fixed points** (equilibria) and what the Jacobian's eigenvalues say about them:")
        table([{"point": fmt_vec(f.point), "type": f.kind,
                "eigenvalues of J": ", ".join(fmt(v) for v in f.eigenvalues)} for f in fps])
    tip("Each particle obeys dx/dt = F(x, t), integrated with 4th-order Runge–Kutta. For time-dependent fields "
        "the arrows change while particles move, so paths are no longer the streamlines. Try the double gyre, "
        "the Lorenz system (3D) or type your own formulas.")
