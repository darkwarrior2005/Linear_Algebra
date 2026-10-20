"""Vector fields in 2D and 3D, including fields that change with time."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core.fields import VectorField, streamlines
from ..field_ui import Window, field_input, latex_or_none
from ..ui import header, show, tip
from ..viz import plot as P

BACKGROUNDS_2D = ["none", "magnitude |F|", "divergence ∇·F", "curl (∇×F)·k"]


def _background(F: VectorField, w: Window, kind: str, t: float, n: int = 70):
    (xs, ys), (X, Y) = w.mesh(n)
    if kind.startswith("magnitude"):
        Z = np.linalg.norm(F(X, Y, t=t), axis=0)
        return xs, ys, Z, "Viridis", None
    if kind.startswith("divergence"):
        return xs, ys, F.divergence_grid([X, Y], t), "RdBu_r", 0.0
    return xs, ys, F.curl_grid([X, Y], t), "PiYG", 0.0


def _stream_trace_2d(F, w, t, seeds_n, color="rgba(255,255,255,0.55)"):
    (xs, ys), (X, Y) = w.mesh(seeds_n)
    seeds = np.stack([X.ravel(), Y.ravel()], axis=1)
    pad = 0.05 * w.size
    lines = streamlines(F, seeds, t, length=w.size * 0.5, steps=50, bound=None)
    S = P.nan_join([L[w.contains(L, pad)] for L in lines])
    return go.Scatter(x=S[:, 0], y=S[:, 1], mode="lines", line=dict(color=color, width=1.2),
                      name="streamlines", hoverinfo="skip")


def _field_2d(F, w, opts, t_values):
    n = opts["density"]
    (xs, ys), (X, Y) = w.mesh(n)
    cell = w.size / n
    mags = [np.nanmax(np.linalg.norm(F(X, Y, t=t), axis=0)) for t in t_values]
    cmax = float(np.nanmax(mags)) if np.isfinite(mags).any() else 1.0
    bg_range = None
    if opts["bg"] != "none":
        vals = [_background(F, w, opts["bg"], t)[2] for t in t_values]
        flat = np.concatenate([v[np.isfinite(v)].ravel() for v in vals]) if vals else np.zeros(1)
        hi = float(np.percentile(np.abs(flat), 98)) if flat.size else 1.0
        bg_range = (-hi, hi) if opts["bg"] != "magnitude |F|" else (0, float(np.percentile(flat, 98)))

    def frame(t):
        tr = []
        if opts["bg"] != "none":
            bx, by, Z, cs, mid = _background(F, w, opts["bg"], t)
            tr.append(go.Heatmap(x=bx, y=by, z=Z, colorscale=cs, zmin=bg_range[0], zmax=bg_range[1], opacity=0.65,
                                 colorbar=dict(title=opts["bg"].split()[0], x=1.02, thickness=12, len=0.6, y=0.25),
                                 hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>value=%{z:.3f}<extra></extra>"))
        if opts["arrows"]:
            U, V = F(X, Y, t=t)
            tr += P.quiver_2d(X, Y, U, V, cell * 0.85, normalise=opts["normalise"], cmax=cmax)
        if opts["stream"]:
            tr.append(_stream_trace_2d(F, w, t, opts["seeds"]))
        return tr

    lay = P.layout_2d(w.layout_range(), 680)
    if len(t_values) == 1:
        return go.Figure(frame(t_values[0]), lay)
    return P.animated_figure(frame, t_values, lay, duration=80, redraw=opts["bg"] != "none")


def _field_3d(F, w, opts, t_values):
    n = opts["density"]
    (xs, ys, zs), (X, Y, Z) = w.mesh(n)
    rng = np.random.default_rng(1)
    seeds = np.stack([rng.uniform(w.bounds[2 * k], w.bounds[2 * k + 1], opts["seeds"] ** 2) for k in range(3)], axis=1)
    mags = [np.nanmax(np.linalg.norm(F(X, Y, Z, t=t), axis=0)) for t in t_values]
    cmax = float(np.nanmax(mags)) if np.isfinite(mags).any() else 1.0

    def frame(t):
        tr = []
        if opts["arrows"]:
            U, V, W = F(X, Y, Z, t=t)
            if opts["normalise"]:
                m = np.sqrt(U ** 2 + V ** 2 + W ** 2)
                with np.errstate(all="ignore"):
                    s = np.where(m > 1e-12, (0.35 + 0.65 * np.clip(m / cmax, 0, 1)) / m, 0)
                cone = P.cone_field_3d(X, Y, Z, U * s, V * s, W * s, sizeref=0.5, cmax=1)
                cone.update(customdata=np.ravel(m), cmin=0.35, cmax=1,
                            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>|F| = %{customdata:.3f}<extra></extra>")
                cone.colorbar.title = "strength"
            else:
                cone = P.cone_field_3d(X, Y, Z, U, V, W, sizeref=0.6, cmax=cmax)
            tr.append(cone)
        if opts["stream"]:
            lines = streamlines(F, seeds, t, length=w.size * 0.8, steps=80)
            S = P.nan_join([L[w.contains(L, 0.05 * w.size)] for L in lines], 3)
            tr.append(go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="lines",
                                   line=dict(color="rgba(255,255,255,0.6)", width=2), name="streamlines",
                                   hoverinfo="skip"))
        return tr

    lay = P.layout_3d(1, 720)
    lay.scene.xaxis.range = list(w.bounds[0:2])
    lay.scene.yaxis.range = list(w.bounds[2:4])
    lay.scene.zaxis.range = list(w.bounds[4:6])
    if len(t_values) == 1:
        return go.Figure(frame(t_values[0]), lay)
    return P.animated_figure(frame, t_values, lay, duration=120, redraw=True)


def symbolic_panel(F: VectorField) -> None:
    sym = F.symbolic()
    if sym is None:
        st.caption("No symbolic form for this expression — derivatives are computed numerically.")
        return
    lx = latex_or_none
    st.latex(r"\nabla\cdot\mathbf F = " + lx(sym["div"]))
    if F.dim == 2:
        st.latex(r"(\nabla\times\mathbf F)\cdot\hat k = " + lx(sym["curl"]))
    else:
        st.latex(r"\nabla\times\mathbf F = " + lx(sym["curl"].T))
    st.latex(r"J = \frac{\partial(F)}{\partial(x)} = " + lx(sym["J"]))


def render() -> None:
    header("Vector fields", "An arrow attached to every point of space — wind, water, forces, velocities. "
                            "Type your own F(x, y, z, t).")
    dim = st.radio("Space", ["2D", "3D"], horizontal=True, key="fld_dim")
    dim = 2 if dim == "2D" else 3
    left, right = st.columns([1, 2.3])
    with left:
        F, w, name = field_input("fld", dim)
        if F is None:
            return
        with st.expander("Display", expanded=True):
            arrows = st.checkbox("Arrows", True, key="fld_arrows")
            normalise = st.checkbox("Equal-length arrows (colour = strength)", True, key="fld_norm")
            stream = st.checkbox("Streamlines", dim == 2, key="fld_stream")
            density = st.slider("Arrow density", 6 if dim == 3 else 8, 14 if dim == 3 else 40,
                                8 if dim == 3 else 22, key=f"fld_density{dim}")
            seeds = st.slider("Streamline seeds per side", 3, 20, 10 if dim == 2 else 5, key=f"fld_seeds{dim}")
            bg = st.selectbox("Background colour", BACKGROUNDS_2D, key="fld_bg") if dim == 2 else "none"
        if F.time_dependent:
            st.success("This field depends on time t.")
            animate = st.checkbox("Animate over time", True, key="fld_anim")
            if animate:
                t0, t1 = st.slider("time range", 0.0, 30.0, (0.0, 2 * np.pi), 0.1, key="fld_trange")
                frames = st.slider("frames", 10, 60, 30, key="fld_frames")
                t_values = list(np.linspace(t0, t1, frames))
            else:
                t_values = [st.slider("time t", 0.0, 30.0, 0.0, 0.05, key="fld_t")]
        else:
            t_values = [0.0]
    opts = dict(arrows=arrows, normalise=normalise, stream=stream, density=density, seeds=seeds, bg=bg)
    fig = _field_2d(F, w, opts, t_values) if dim == 2 else _field_3d(F, w, opts, t_values)
    with right:
        show(fig, key="fld_fig")
    with st.expander("Formulas: divergence, curl and Jacobian", expanded=True):
        symbolic_panel(F)
    tip("Arrows show direction (and colour shows strength). Streamlines follow the arrows like smoke in a wind "
        "tunnel. Red/blue divergence = sources/sinks; curl colour = local spinning. Time-dependent fields "
        "change shape as t runs — press ▶ Play.  See *Flows & particles* to release particles into the field.")
