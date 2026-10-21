"""Vector functions of time r(t): position, velocity, acceleration and the moving frame."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core.fields import ParametricCurve
from ..core.parsing import ExpressionError, eval_number
from ..presets import CURVES
from ..ui import dimension_picker, fmt, fmt_vec, header, show, tip
from ..viz import plot as P


def _curve_input(dim: int):
    presets = CURVES[dim]
    names = list(presets) + ["(custom)"]
    pkey = f"mot_preset{dim}"

    def apply() -> None:
        name = st.session_state[pkey]
        if name in presets:
            comps, (a, b) = presets[name]
            for k, c in enumerate(comps):
                st.session_state[f"mot_c{dim}_{k}"] = c
            st.session_state[f"mot_a{dim}"] = str(a)
            st.session_state[f"mot_b{dim}"] = str(b)

    if pkey not in st.session_state or f"mot_c{dim}_0" not in st.session_state:
        st.session_state[pkey] = names[0]
        apply()
    st.selectbox("Curve preset", names, key=pkey, on_change=apply)
    comps = [st.text_input(f"{'xyz'[k]}(t) =", key=f"mot_c{dim}_{k}") for k in range(dim)]
    c1, c2 = st.columns(2)
    a = c1.text_input("t from", key=f"mot_a{dim}")
    b = c2.text_input("t to", key=f"mot_b{dim}")
    try:
        curve = ParametricCurve(comps)
        t0, t1 = eval_number(a), eval_number(b)
        if t1 <= t0:
            raise ExpressionError("'t to' must be larger than 't from'")
        curve(np.array([t0]))
    except (ExpressionError, ValueError, TypeError) as exc:
        st.error(str(exc))
        return None, 0.0, 1.0
    return curve, t0, t1


def render() -> None:
    header("Motion along a curve  r(t)",
           "A vector that changes with time traces a path. Its derivatives are velocity and acceleration.")
    dim = dimension_picker("mot_dim", default=3)
    left, right = st.columns([1, 2.3])
    with left:
        curve, t0, t1 = _curve_input(dim)
        if curve is None:
            return
        show_v = st.checkbox("Velocity r′(t)", True, key="mot_v")
        show_a = st.checkbox("Acceleration r″(t)", True, key="mot_a")
        show_frame = st.checkbox("Frenet frame T, N, B (unit vectors)", dim == 3, key="mot_tnb")
        show_osc = st.checkbox("Osculating circle (best-fitting circle)", dim == 2, key="mot_osc")
        scale = st.slider("Arrow scale", 0.05, 3.0, 0.5, 0.05, key="mot_scale")
        frames = st.slider("Animation frames", 20, 150, 80, key="mot_frames")
    ts = np.linspace(t0, t1, 600)
    K = curve.kinematics(ts)
    R = K["r"]
    finite = R[np.isfinite(R).all(axis=1)]
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    pad = 0.15 * max(float(np.max(hi - lo)), 1e-6) + scale
    tf = np.linspace(t0, t1, frames)
    Kf = curve.kinematics(tf)
    head = (float(np.max(hi - lo)) + 2 * pad) * 0.025

    def frame(k):
        k = int(k)
        r = Kf["r"][k]
        tr = []
        if dim == 2:
            tr.append(P.points_2d(r, "#ffffff", "position r(t)", 12))
            tr += P.arrows_2d(r, color="rgba(255,255,255,0.35)", name="r(t) from origin", head=head, width=1.5)
            tr += P.arrows_2d(Kf["v"][k] * scale if show_v else np.zeros((0, 2)), r, color=P.color(0),
                              name="velocity", head=head)
            tr += P.arrows_2d(Kf["a"][k] * scale * scale if show_a else np.zeros((0, 2)), r, color=P.color(4),
                              name="acceleration", head=head)
            kap = Kf["curvature"][k]
            if show_osc and np.isfinite(kap) and kap > 1e-6 and 1 / kap < 1e3:
                c = r + Kf["N"][k][:2] / kap
                circ = c[:, None] + P.circle_points(120, 1 / kap)
                tr.append(P.curve_2d(circ, "rgba(34,184,207,0.8)", "osculating circle", 1.5, "dot"))
            else:
                tr.append(go.Scatter(x=[None], y=[None], mode="lines", showlegend=False))
        else:
            tr.append(P.points_3d(r, "#ffffff", "position r(t)", 7))
            tr += P.arrows_3d(Kf["v"][k] * scale if show_v else np.zeros((0, 3)), r, color=P.color(0),
                              name="velocity", head=head * 1.6)
            tr += P.arrows_3d(Kf["a"][k] * scale * scale if show_a else np.zeros((0, 3)), r, color=P.color(4),
                              name="acceleration", head=head * 1.6)
            L = scale * 2
            for vec, colr, lab in ((Kf["T"][k], P.I_COLOR, "T"), (Kf["N"][k], P.J_COLOR, "N"), (Kf["B"][k], P.K_COLOR, "B")):
                ok = show_frame and np.all(np.isfinite(vec))
                tr += P.arrows_3d(vec * L if ok else np.zeros((0, 3)), r, color=colr, name=f"{lab} (unit)",
                                  head=head * 1.4, labels=[lab] if ok else None)
        return tr

    if dim == 2:
        static = [go.Scatter(x=R[:, 0], y=R[:, 1], mode="lines", line=dict(color="rgba(77,171,247,0.8)", width=3),
                             name="path")]
        lay = P.layout_2d([lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad], 640)
    else:
        sp = K["speed"]
        lo_s, hi_s = float(np.nanmin(sp)), float(np.nanmax(sp))
        if hi_s - lo_s < 1e-6 * max(1.0, hi_s):  # constant speed: widen so the colour bar stays readable
            lo_s, hi_s = lo_s - 0.5, hi_s + 0.5
        static = [go.Scatter3d(x=R[:, 0], y=R[:, 1], z=R[:, 2], mode="lines",
                               line=dict(color=sp, colorscale="Viridis", width=6, cmin=lo_s, cmax=hi_s,
                                         colorbar=dict(title="speed", thickness=12, len=0.5)), name="path")]
        lay = P.layout_3d(1, 680)
        c = (lo + hi) / 2
        half = float(np.max(hi - lo)) / 2 + pad
        lay.scene.xaxis.range = [c[0] - half, c[0] + half]
        lay.scene.yaxis.range = [c[1] - half, c[1] + half]
        lay.scene.zaxis.range = [c[2] - half, c[2] + half]
    fig = P.animated_figure(frame, range(frames), lay, static, duration=60, initial=0,
                            label=lambda k: f"{tf[int(k)]:.2f}", redraw=dim == 3)
    with right:
        show(fig, key="mot_fig")
    with left:
        tq = st.slider("Inspect at time", float(t0), float(t1), float(t0 + (t1 - t0) / 3), (t1 - t0) / 500, key="mot_tq")
    q = curve.kinematics(np.array([tq]))
    c = st.columns(5)
    c[0].metric("position", fmt_vec(q["r"][0]))
    c[1].metric("velocity", fmt_vec(q["v"][0]))
    c[2].metric("speed |r′|", fmt(q["speed"][0]))
    c[3].metric("curvature κ", fmt(q["curvature"][0]))
    c[4].metric("torsion τ" if dim == 3 else "radius 1/κ", fmt(q["torsion"][0]) if dim == 3 else
                (fmt(1 / q["curvature"][0]) if q["curvature"][0] > 1e-9 else "∞"))
    g = go.Figure([go.Scatter(x=ts, y=K["speed"], name="speed", line=dict(color=P.color(0))),
                   go.Scatter(x=ts, y=np.clip(K["curvature"], 0, np.nanpercentile(K["curvature"], 99) * 1.2 + 1e-9),
                              name="curvature", line=dict(color=P.color(3)))],
                  P.layout_2d([t0, t1, 0, 1], 260))
    g.update_yaxes(scaleanchor=None, autorange=True)
    g.update_layout(xaxis_title="t", showlegend=True)
    g.add_vline(x=tq, line_color="#ffffff", line_dash="dot")
    show(g, key="mot_graph")
    tip("r(t) is a vector that changes with time. Velocity r′(t) is tangent to the path; acceleration r″(t) "
        "points toward the inside of the bend when the speed is constant. T, N, B form a little moving "
        "coordinate system — an orthonormal basis that rides along the curve.")
