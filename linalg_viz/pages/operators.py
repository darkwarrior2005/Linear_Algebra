"""Differential operators you can see: gradient, divergence, curl, Laplacian and the Jacobian."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..core.fields import VectorField, field_rhs, rk4_paths
from ..field_ui import Window, field_input, latex_or_none, scalar_input
from ..ui import fmt, fmt_vec, header, latex_matrix, show, tip
from ..viz import plot as P

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2 only has trapz


def _probe(prefix: str, w: Window, default=(0.6, 0.4, 0.3)) -> np.ndarray:
    vals = []
    for k, axis in enumerate("xyz"[: w.dim]):
        lo, hi = w.bounds[2 * k], w.bounds[2 * k + 1]
        d = float(np.clip(default[k] * (hi - lo) / 2 + (hi + lo) / 2, lo, hi))
        vals.append(st.slider(f"probe {axis}", float(lo), float(hi), d, (hi - lo) / 200, key=f"{prefix}_p{axis}{w.dim}"))
    return np.array(vals)


def _time(prefix: str, needed: bool) -> float:
    return st.slider("time t", 0.0, 20.0, 0.0, 0.05, key=f"{prefix}_time") if needed else 0.0


def _scalar_window(dim: int, span: float = 2.5) -> Window:
    return Window(tuple([-span, span] * dim))


def _quiver_light(F: VectorField, w: Window, t: float, n: int = 18):
    (_, _), (X, Y) = w.mesh(n)
    U, V = F(X, Y, t=t)
    return P.quiver_2d(X, Y, U, V, w.size / n * 0.8, colorscale="Greys")[:-1]


def _circle(p, r, n=240):
    th = np.linspace(0, 2 * np.pi, n)
    return np.stack([p[0] + r * np.cos(th), p[1] + r * np.sin(th)], axis=1), th


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def _gradient() -> None:
    dim = 2 if st.radio("Space", ["2D", "3D"], horizontal=True, key="grad_dim") == "2D" else 3
    left, right = st.columns([1, 2.3])
    with left:
        f = scalar_input("grad", dim)
        if f is None:
            return
        span = st.slider("Window half-size", 0.5, 10.0, 2.5, 0.25, key=f"grad_span{dim}")
        w = _scalar_window(dim, span)
        t = _time("grad", "t" in f.expr.uses)
        p = _probe("grad", w)
        if dim == 2:
            ang = st.slider("Direction for directional derivative (°)", 0.0, 360.0, 0.0, 1.0, key="grad_ang")
    n = 60 if dim == 2 else 14
    axes, mesh = w.mesh(n)
    gp = f.gradient_grid([np.array(c) for c in p], t).ravel()
    fp = float(f(*p, t=t))
    if dim == 2:
        xs, ys = axes
        Z = f(*mesh, t=t)
        tr = [go.Contour(x=xs, y=ys, z=Z, colorscale="Viridis", contours=dict(showlabels=True), opacity=0.85,
                         line=dict(width=1), colorbar=dict(title="f", thickness=12, len=0.6),
                         hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>f=%{z:.3f}<extra></extra>")]
        (_, _), (Xq, Yq) = w.mesh(16)
        Gq = f.gradient_grid([Xq, Yq], t)
        tr += P.quiver_2d(Xq, Yq, Gq[0], Gq[1], w.size / 16 * 0.8, colorscale="Reds")[:-1]
        head = w.size * 0.03
        gn = np.linalg.norm(gp)
        scale = (w.size * 0.18 / gn) if gn > 1e-12 else 0
        tr += P.arrows_2d(gp * scale, p, color="#ffffff", name="∇f (steepest uphill)", head=head, width=4)
        if gn > 1e-12:
            tang = np.array([-gp[1], gp[0]]) / gn * w.size * 0.15
            tr.append(go.Scatter(x=[p[0] - tang[0], p[0] + tang[0]], y=[p[1] - tang[1], p[1] + tang[1]], mode="lines",
                                 line=dict(color="#ffffff", dash="dash"), name="level curve direction (∇f ⊥ it)"))
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        tr += P.arrows_2d(u * w.size * 0.12, p, color=P.color(0), name="direction u", head=head)
        tr.append(P.points_2d(p, "#ffffff", "", 9, False))
        fig = go.Figure(tr, P.layout_2d(w.layout_range(), 560))
        surf = go.Figure([go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis", opacity=0.9, showscale=False),
                          go.Scatter3d(x=[p[0]], y=[p[1]], z=[fp], mode="markers", marker=dict(size=6, color="white"),
                                       name="probe")],
                         P.layout_3d(span, 560))
        zr = [float(np.nanmin(Z)), float(np.nanmax(Z))]
        surf.update_layout(scene=dict(zaxis=dict(range=zr), aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6)))
        if gn > 1e-12:
            steps = np.linspace(0, w.size * 0.2, 12)
            path = p[None, :] + steps[:, None] * gp / gn
            surf.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=f(path[:, 0], path[:, 1], t=t), mode="lines",
                                        line=dict(color="white", width=7), name="uphill along ∇f"))
        with right:
            a, b = st.columns(2)
            with a:
                show(fig, key="grad_fig2")
            with b:
                show(surf, key="grad_surf")
        c = st.columns(4)
        c[0].metric("f(p)", fmt(fp))
        c[1].metric("∇f(p)", fmt_vec(gp))
        c[2].metric("|∇f| = steepest slope", fmt(np.linalg.norm(gp)))
        c[3].metric("D_u f = ∇f·u", fmt(float(np.dot(gp, u))))
    else:
        X, Y, Zg = mesh
        V = f(X, Y, Zg, t=t)
        tr = [go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Zg.ravel(), value=V.ravel(), surface_count=4, opacity=0.25,
                            colorscale="Viridis", caps=dict(x_show=False, y_show=False, z_show=False),
                            colorbar=dict(title="f", thickness=12, len=0.6))]
        (_, _, _), (Xc, Yc, Zc) = w.mesh(7)
        Gc = f.gradient_grid([Xc, Yc, Zc], t)
        cone = P.cone_field_3d(Xc, Yc, Zc, *Gc, sizeref=0.6, colorscale="Reds")
        cone.showscale = False
        tr.append(cone)
        tr += P.arrows_3d(gp / (np.linalg.norm(gp) or 1) * span * 0.4, p, color="#ffffff", name="∇f at probe",
                          head=span * 0.08)
        with right:
            show(go.Figure(tr, P.layout_3d(span, 640)), key="grad_fig3")
        c = st.columns(3)
        c[0].metric("f(p)", fmt(fp))
        c[1].metric("∇f(p)", fmt_vec(gp))
        c[2].metric("|∇f|", fmt(np.linalg.norm(gp)))
    sym = f.symbolic()
    if sym:
        st.latex(r"\nabla f = " + latex_or_none(sym["grad"].T) + r",\qquad \nabla^2 f = " + latex_or_none(sym["lap"]))
        st.latex(r"\text{Hessian } H = " + latex_or_none(sym["hess"]))
    tip("The gradient points straight uphill and is perpendicular to the level curves/surfaces. Its length is "
        "the steepness. The slope in any other direction u is ∇f·u — the dot product again!")


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def _divergence() -> None:
    dim = 2 if st.radio("Space", ["2D", "3D"], horizontal=True, key="div_dim") == "2D" else 3
    left, right = st.columns([1, 2.3])
    with left:
        F, w, _ = field_input("div", dim, "Source" if dim == 2 else "Source")
        if F is None:
            return
        t = _time("div", F.time_dependent)
        p = _probe("div", w)
        r = st.slider("Probe radius", 0.05, w.size / 4, w.size / 10, w.size / 200, key=f"div_r{dim}")
    divp = F.divergence_at(p, t)
    if dim == 2:
        (xs, ys), (X, Y) = w.mesh(70)
        D = F.divergence_grid([X, Y], t)
        hi = float(np.nanpercentile(np.abs(D), 98)) or 1.0
        tr = [go.Heatmap(x=xs, y=ys, z=D, colorscale="RdBu_r", zmin=-hi, zmax=hi, opacity=0.7,
                         colorbar=dict(title="∇·F", thickness=12, len=0.6))]
        tr += _quiver_light(F, w, t)
        C, th = _circle(p, r, 400)
        tr.append(go.Scatter(x=C[:, 0], y=C[:, 1], mode="lines", line=dict(color="white", width=2), name="probe circle"))
        k = 16
        thk = np.linspace(0, 2 * np.pi, k, endpoint=False)
        pts = np.stack([p[0] + r * np.cos(thk), p[1] + r * np.sin(thk)], axis=1)
        nrm = np.stack([np.cos(thk), np.sin(thk)], axis=1)
        Fv = F(pts[:, 0], pts[:, 1], t=t).T
        fn = np.sum(Fv * nrm, axis=1)
        L = r * 0.9 / (np.max(np.abs(fn)) or 1)
        out, inn = fn >= 0, fn < 0
        tr += P.arrows_2d((nrm * fn[:, None] * L)[out], pts[out], color="#51cf66", name="flowing out", head=r * 0.25)
        tr += P.arrows_2d((nrm * fn[:, None] * L)[inn], pts[inn], color="#ff6b6b", name="flowing in", head=r * 0.25)
        Fc = F(C[:, 0], C[:, 1], t=t).T
        flux = float(_trapz(np.sum(Fc * np.stack([np.cos(th), np.sin(th)], 1), 1) * r, th))
        ratio = flux / (np.pi * r * r)
        with right:
            show(go.Figure(tr, P.layout_2d(w.layout_range(), 640)), key="div_fig")
        c = st.columns(3)
        c[0].metric("∇·F at probe", fmt(divp))
        c[1].metric("net outflow through circle", fmt(flux))
        c[2].metric("outflow ÷ area", fmt(ratio), help="Shrink the radius: this tends to the divergence.")
    else:
        (_, _, _), (X, Y, Z) = w.mesh(16)
        D = F.divergence_grid([X, Y, Z], t)
        hi = float(np.nanpercentile(np.abs(D), 98)) or 1.0
        tr = [go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=D.ravel(), isomin=-hi, isomax=hi, opacity=0.12,
                        surface_count=12, colorscale="RdBu_r", colorbar=dict(title="∇·F", thickness=12, len=0.6))]
        (_, _, _), (Xc, Yc, Zc) = w.mesh(6)
        cone = P.cone_field_3d(Xc, Yc, Zc, *F(Xc, Yc, Zc, t=t), sizeref=0.6, colorscale="Greys")
        cone.showscale = False
        tr.append(cone)
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
        S = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
        pts = p[:, None, None] + r * S
        Fs = F(*pts, t=t)
        fn = np.sum(Fs * S, axis=0)
        flux = float(np.sum(fn * np.sin(v)) * r * r * (2 * np.pi / 39) * (np.pi / 19))
        tr.append(go.Surface(x=pts[0], y=pts[1], z=pts[2], surfacecolor=fn, colorscale="RdYlGn", opacity=0.9,
                             showscale=False, name="probe sphere (green = out)"))
        lay = P.layout_3d(1, 680)
        lay.scene.xaxis.range, lay.scene.yaxis.range, lay.scene.zaxis.range = (list(w.bounds[0:2]),
                                                                               list(w.bounds[2:4]), list(w.bounds[4:6]))
        with right:
            show(go.Figure(tr, lay), key="div_fig3")
        c = st.columns(3)
        c[0].metric("∇·F at probe", fmt(divp))
        c[1].metric("net outflow through sphere", fmt(flux))
        c[2].metric("outflow ÷ volume", fmt(flux / (4 / 3 * np.pi * r ** 3)))
    sym = F.symbolic()
    if sym:
        st.latex(r"\nabla\cdot\mathbf F = " + latex_or_none(sym["div"]))
    tip("Divergence measures how much stuff is created (red/positive: a source) or swallowed (blue/negative: "
        "a sink) at a point. Put the probe anywhere: green arrows leave the circle, red ones enter. Net outflow "
        "divided by the area approaches the divergence as the circle shrinks — the divergence theorem in miniature.")


# ---------------------------------------------------------------------------
# Curl
# ---------------------------------------------------------------------------

def _curl() -> None:
    dim = 2 if st.radio("Space", ["2D", "3D"], horizontal=True, key="curl_dim") == "2D" else 3
    left, right = st.columns([1, 2.3])
    with left:
        F, w, _ = field_input("curl", dim, "Shear flow" if dim == 2 else "Rotation about z")
        if F is None:
            return
        t = _time("curl", F.time_dependent)
        p = _probe("curl", w)
        r = st.slider("Paddle-wheel radius", 0.05, w.size / 4, w.size / 10, w.size / 200, key=f"curl_r{dim}")
    curlp = F.curl_at(p, t)
    if dim == 2:
        (xs, ys), (X, Y) = w.mesh(70)
        Cg = F.curl_grid([X, Y], t)
        hi = float(np.nanpercentile(np.abs(Cg), 98)) or 1.0
        static = [go.Heatmap(x=xs, y=ys, z=Cg, colorscale="PiYG", zmin=-hi, zmax=hi, opacity=0.7,
                             colorbar=dict(title="curl", thickness=12, len=0.6))]
        static += _quiver_light(F, w, t)
        C, th = _circle(p, r, 400)
        static.append(go.Scatter(x=C[:, 0], y=C[:, 1], mode="lines", line=dict(color="white", width=1.5),
                                 name="paddle wheel"))
        omega = curlp / 2

        def frame(s):
            ang = omega * s + np.arange(4) * np.pi / 2
            xs_, ys_ = [], []
            for a in ang:
                xs_ += [p[0], p[0] + r * np.cos(a), None]
                ys_ += [p[1], p[1] + r * np.sin(a), None]
            return [go.Scatter(x=xs_, y=ys_, mode="lines+markers", line=dict(color="#ffd43b", width=4),
                               marker=dict(size=7), name=f"spins at ω = curl/2 = {fmt(omega)}")]

        fig = P.animated_figure(frame, np.linspace(0, 2 * np.pi, 60), P.layout_2d(w.layout_range(), 640), static,
                                duration=60, label=lambda s: f"{s:.2f}", initial=0, slider_prefix="time ")
        Fc = F(C[:, 0], C[:, 1], t=t).T
        tang = np.stack([-np.sin(th), np.cos(th)], 1)
        circ = float(_trapz(np.sum(Fc * tang, 1) * r, th))
        with right:
            show(fig, key="curl_fig")
        c = st.columns(3)
        c[0].metric("curl at probe", fmt(curlp))
        c[1].metric("circulation around circle", fmt(circ))
        c[2].metric("circulation ÷ area", fmt(circ / (np.pi * r * r)))
    else:
        (_, _, _), (X, Y, Z) = w.mesh(6)
        Cg = F.curl_grid([X, Y, Z], t)
        tr = []
        cf = P.cone_field_3d(X, Y, Z, *F(X, Y, Z, t=t), sizeref=0.6, colorscale="Greys", name="F")
        cf.showscale = False
        tr.append(cf)
        cc = P.cone_field_3d(X, Y, Z, *Cg, sizeref=0.6, colorscale="PiYG", name="curl F")
        cc.colorbar.title = "|curl|"
        tr.append(cc)
        cn = np.linalg.norm(curlp)
        if cn > 1e-9:
            axis = curlp / cn
            a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = np.cross(axis, a)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            th = np.linspace(0, 2 * np.pi, 60)
            ring = p[:, None] + r * (np.outer(e1, np.cos(th)) + np.outer(e2, np.sin(th)))
            tr.append(go.Scatter3d(x=ring[0], y=ring[1], z=ring[2], mode="lines", line=dict(color="#ffd43b", width=6),
                                   name="paddle wheel (spins about curl)"))
            tr += P.arrows_3d(axis * w.size * 0.2, p, color="#ffd43b", name="curl direction (spin axis)",
                              head=w.size * 0.04)
        lay = P.layout_3d(1, 680)
        lay.scene.xaxis.range, lay.scene.yaxis.range, lay.scene.zaxis.range = (list(w.bounds[0:2]),
                                                                               list(w.bounds[2:4]), list(w.bounds[4:6]))
        with right:
            show(go.Figure(tr, lay), key="curl_fig3")
        c = st.columns(2)
        c[0].metric("curl F at probe", fmt_vec(curlp))
        c[1].metric("spin rate |curl|/2", fmt(np.linalg.norm(curlp) / 2))
    sym = F.symbolic()
    if sym:
        st.latex((r"(\nabla\times\mathbf F)\cdot\hat k = " if dim == 2 else r"\nabla\times\mathbf F = ")
                 + latex_or_none(sym["curl"] if dim == 2 else sym["curl"].T))
    tip("Curl measures local spinning. Imagine a tiny paddle wheel dropped into the flow: it turns at half the "
        "curl. Note that a shear flow (no visible swirl!) still spins the wheel, while a point vortex has zero "
        "curl away from its centre.")


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def _laplacian() -> None:
    left, right = st.columns([1, 2.3])
    with left:
        f = scalar_input("lap", 2)
        if f is None:
            return
        span = st.slider("Window half-size", 0.5, 10.0, 2.5, 0.25, key="lap_span")
        w = _scalar_window(2, span)
        t = _time("lap", "t" in f.expr.uses)
        p = _probe("lap", w)
        r = st.slider("Neighbourhood radius", 0.05, span / 2, span / 8, span / 100, key="lap_r")
    (xs, ys), (X, Y) = w.mesh(80)
    Z = f(X, Y, t=t)
    L = f.laplacian_grid([X, Y], t)
    hi = float(np.nanpercentile(np.abs(L), 98)) or 1.0
    C, th = _circle(p, r, 200)
    with right:
        a, b = st.columns(2)
        with a:
            fig = go.Figure([go.Heatmap(x=xs, y=ys, z=Z, colorscale="Viridis", colorbar=dict(title="f", len=0.5)),
                             go.Scatter(x=C[:, 0], y=C[:, 1], mode="lines", line=dict(color="white"), name="neighbours")],
                            P.layout_2d(w.layout_range(), 480, "f"))
            show(fig, key="lap_f")
        with b:
            fig = go.Figure([go.Heatmap(x=xs, y=ys, z=L, colorscale="RdBu_r", zmin=-hi, zmax=hi,
                                        colorbar=dict(title="∇²f", len=0.5)),
                             go.Scatter(x=C[:, 0], y=C[:, 1], mode="lines", line=dict(color="black"), name="neighbours")],
                            P.layout_2d(w.layout_range(), 480, "∇²f"))
            show(fig, key="lap_L")
    avg = float(np.mean(f(C[:-1, 0], C[:-1, 1], t=t)))
    fp = float(f(*p, t=t))
    lp = float(f.laplacian_grid([np.array(p[0]), np.array(p[1])], t))
    c = st.columns(4)
    c[0].metric("f(p)", fmt(fp))
    c[1].metric("average of neighbours", fmt(avg))
    c[2].metric("∇²f(p)", fmt(lp))
    c[3].metric("4·(avg − f(p)) / r²", fmt(4 * (avg - fp) / r ** 2), help="Approaches ∇²f as r shrinks.")
    tip("The Laplacian compares a point with the average of its neighbours: positive where the point is lower "
        "than its surroundings (a valley), negative at a peak. The heat equation ∂f/∂t = ∇²f smooths things out "
        "for exactly this reason.")


# ---------------------------------------------------------------------------
# Jacobian: the linear algebra inside a field
# ---------------------------------------------------------------------------

def _jacobian() -> None:
    left, right = st.columns([1, 2.3])
    with left:
        F, w, _ = field_input("jac", 2, "Spiral sink")
        if F is None:
            return
        t = _time("jac", F.time_dependent)
        p = _probe("jac", w)
        h = st.slider("Zoom (neighbourhood size)", 0.05, w.size / 3, w.size / 8, w.size / 200, key="jac_h")
        dt = st.slider("Let the square flow for time", 0.0, 1.0, 0.25, 0.01, key="jac_dt")
    J = F.jacobian(p, t)
    Fp = F.at(p, t)
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * h / 2
    # fine square outline (for the true flow)
    edge = np.concatenate([np.linspace(sq[k], sq[(k + 1) % 4], 20, endpoint=False) for k in range(4)])
    if dt > 0:
        _, paths = rk4_paths(field_rhs(F, t), np.vstack([p, p + edge]), 0, dt, 30)
        moved_true = paths[-1][1:] - paths[-1][0]
    else:
        moved_true = edge
    moved_lin = edge @ la.matrix_exp(J * dt).T
    sw = h * 1.6
    tr = []
    (_, _), (X, Y) = Window((p[0] - sw, p[0] + sw, p[1] - sw, p[1] + sw)).mesh(11)
    U, V = F(X, Y, t=t)
    tr += P.quiver_2d(X - p[0], Y - p[1], U - Fp[0], V - Fp[1], sw / 11 * 0.8, colorscale="Greys")[:-1]
    tr.append(P.polygon_2d(edge, "rgba(255,255,255,0.6)", "small square", 0.05))
    tr.append(P.polygon_2d(moved_true, "#ffd43b", "after flowing (true)", 0.25))
    tr.append(P.polygon_2d(moved_lin, "#4dabf7", "after flowing (linear: e^{J·t})", 0.15))
    for p_ in la.eigen_analysis(J):
        if p_.is_real:
            tr.append(P.line_through_origin_2d(np.real(p_.vector), sw, P.color(4), f"eigenvector of J, λ = {fmt(p_.value.real)}"))
    lay = P.layout_2d([-sw, sw, -sw, sw], 620, "Zoomed in at the probe (moving with the flow)")
    with right:
        show(go.Figure(tr, lay), key="jac_fig")
    S = (J + J.T) / 2
    Wm = (J - J.T) / 2
    c1, c2, c3 = st.columns(3)
    with c1:
        latex_matrix(J, "J = ")
    with c2:
        latex_matrix(S, r"\tfrac{J+J^T}{2} = ")
        st.caption("Stretching part (strain). Its trace is the divergence.")
    with c3:
        latex_matrix(Wm, r"\tfrac{J-J^T}{2} = ")
        st.caption("Spinning part (rotation). Its entry is ±curl/2.")
    st.markdown(f"div F = tr J = **{fmt(np.trace(J))}**, curl F = J₂₁ − J₁₂ = **{fmt(J[1, 0] - J[0, 1])}**, "
                f"det J = **{fmt(np.linalg.det(J))}** (area scaling of the local map), "
                f"eigenvalues: {', '.join(fmt(v) for v in np.linalg.eigvals(J))}")
    tip("Zoom in far enough and every smooth field looks linear: F(p + d) ≈ F(p) + J d. The Jacobian matrix J "
        "is that linear map, so everything from the linear-algebra pages applies: its eigenvectors are the "
        "directions the little square gets stretched along, its determinant the area change, and it splits "
        "into a symmetric stretch plus a rotation.")


def _identities() -> None:
    st.markdown("Two famous identities, checked numerically on a grid for **your** functions:")
    f = scalar_input("idn", 3)
    F, w, _ = field_input("idn", 3, "ABC flow")
    if f is None or F is None:
        return
    (_, _, _), mesh = w.mesh(9)
    h = 1e-3
    G = f.gradient_grid(mesh, 0.0, h)
    # curl(grad f) via nested differences
    def g(k):
        return lambda *xyz: f.gradient_grid(list(xyz), 0.0, h)[k]
    d = lambda comp, var: (g(comp)(*[m + (h if i == var else 0) for i, m in enumerate(mesh)])  # noqa: E731
                           - g(comp)(*[m - (h if i == var else 0) for i, m in enumerate(mesh)])) / (2 * h)
    curl_grad = np.stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])
    Cf = F.curl_grid(mesh, 0.0)
    div_curl = np.zeros_like(mesh[0])
    for k in range(3):
        plus = [m + (h if i == k else 0) for i, m in enumerate(mesh)]
        minus = [m - (h if i == k else 0) for i, m in enumerate(mesh)]
        div_curl += (F.curl_grid(plus, 0.0)[k] - F.curl_grid(minus, 0.0)[k]) / (2 * h)
    c = st.columns(2)
    c[0].metric("max |curl(∇f)|", f"{np.nanmax(np.abs(curl_grad)):.2e}", help="Should be ≈ 0 (rounding only).")
    c[1].metric("max |div(curl F)|", f"{np.nanmax(np.abs(div_curl)):.2e}", help="Should be ≈ 0 (rounding only).")
    st.caption(f"Typical size of ∇f: {np.nanmean(np.abs(G)):.3g}; typical size of curl F: {np.nanmean(np.abs(Cf)):.3g}. "
               "Gradient fields never swirl, and swirl fields never have sources.")


def render() -> None:
    header("Differential operators", "∇ acting on fields: gradient, divergence, curl, Laplacian — and the Jacobian "
                                     "matrix that links calculus back to linear algebra.")
    tabs = st.tabs(["⛰️ Gradient ∇f", "💨 Divergence ∇·F", "🌀 Curl ∇×F", "🔥 Laplacian ∇²f", "🔍 Jacobian (local linear map)",
                    "🧪 Identities"])
    with tabs[0]:
        _gradient()
    with tabs[1]:
        _divergence()
    with tabs[2]:
        _curl()
    with tabs[3]:
        _laplacian()
    with tabs[4]:
        _jacobian()
    with tabs[5]:
        _identities()
