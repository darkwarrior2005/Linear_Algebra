"""Plotly building blocks: arrows, grids, shapes, fields and animations in 2D/3D."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go

# Consistent colours across the whole app.
I_COLOR = "#ff6b6b"   # first basis vector (x)
J_COLOR = "#51cf66"   # second basis vector (y)
K_COLOR = "#4dabf7"   # third basis vector (z)
BASIS_COLORS = (I_COLOR, J_COLOR, K_COLOR)
PALETTE = ("#ffd43b", "#cc5de8", "#ff922b", "#22b8cf", "#f06595", "#94d82d", "#e8590c", "#748ffc")
GRID_COLOR = "rgba(77,171,247,0.55)"
GRID_FAINT = "rgba(200,200,200,0.12)"
AXIS_COLOR = "rgba(230,230,230,0.6)"
EIGEN_COLOR = "#ffd43b"
BG = "#0e1117"


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ---------------------------------------------------------------------------
# Fast polyline helpers: NumPy arrays with NaN gaps (much faster for Plotly than
# Python lists with None separators)
# ---------------------------------------------------------------------------

def nan_join(parts: Sequence[np.ndarray], dim: int = 2) -> np.ndarray:
    """Stack several (k, dim) point arrays into one, separated by NaN rows."""
    parts = [np.asarray(p, float) for p in parts if len(p)]
    if not parts:
        return np.full((1, dim), np.nan)
    gap = np.full((1, parts[0].shape[1]), np.nan)
    out = []
    for p in parts:
        out += [p, gap]
    return np.concatenate(out)


def segments(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Rows start, end, NaN for each pair — one polyline with many separate segments."""
    starts = np.asarray(starts, float)
    ends = np.asarray(ends, float)
    if len(starts) == 0:
        return np.full((1, starts.shape[1] if starts.ndim == 2 else 2), np.nan)
    gap = np.full_like(starts, np.nan)
    return np.stack([starts, ends, gap], axis=1).reshape(-1, starts.shape[1])


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

def layout_2d(span: float | Sequence[float] = 5, height: int = 620, title: str | None = None,
              show_axes: bool = True) -> go.Layout:
    if np.isscalar(span):
        xr = yr = [-span, span]
    else:
        xr, yr = [span[0], span[1]], [span[2], span[3]]
    axis = dict(zeroline=show_axes, zerolinecolor=AXIS_COLOR, zerolinewidth=1.5,
                gridcolor=GRID_FAINT, showgrid=True)
    return go.Layout(
        template="plotly_dark", height=height, title=title, paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(range=xr, **axis),
        yaxis=dict(range=yr, scaleanchor="x", scaleratio=1, **axis),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.99),
        uirevision="keep",
    )


def layout_3d(span: float = 4, height: int = 680, title: str | None = None) -> go.Layout:
    ax = lambda name: dict(range=[-span, span], title=name, backgroundcolor=BG,  # noqa: E731
                           gridcolor="rgba(200,200,200,0.15)", zerolinecolor=AXIS_COLOR,
                           showbackground=True)
    return go.Layout(
        template="plotly_dark", height=height, title=title, paper_bgcolor=BG,
        scene=dict(xaxis=ax("x"), yaxis=ax("y"), zaxis=ax("z"), aspectmode="cube",
                   camera=dict(eye=dict(x=1.3, y=1.15, z=0.8))),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.99),
        uirevision="keep",
    )


# ---------------------------------------------------------------------------
# 2D primitives
# ---------------------------------------------------------------------------

def arrows_2d(vectors: np.ndarray, origins: np.ndarray | None = None, color: str = PALETTE[0],
              name: str = "", width: float = 3.5, head: float = 0.22, showlegend: bool = True,
              labels: Sequence[str] | None = None, dash: str | None = None, opacity: float = 1.0) -> list:
    """Arrows from ``origins`` along ``vectors`` (both shape (n, 2)); returns 3 traces.

    The head length is ``head`` in data units, capped at 40 % of each arrow.
    The returned trace count is always 3, which keeps animation frames aligned.
    """
    V = np.asarray(vectors, dtype=float).reshape(-1, 2)
    O = np.zeros_like(V) if origins is None else np.broadcast_to(np.asarray(origins, float), V.shape)
    tips = O + V
    L = np.hypot(V[:, 0], V[:, 1])
    ok = np.isfinite(L) & (L > 1e-9)
    Vk, Ok, Tk, Lk = V[ok], O[ok], tips[ok], L[ok]
    u = Vk / Lk[:, None] if len(Vk) else Vk
    n = np.stack([-u[:, 1], u[:, 0]], axis=1) if len(u) else u
    hl = np.minimum(head, 0.4 * Lk)[:, None]
    base = Tk - u * hl
    shaft = segments(Ok, base)
    if len(Tk):
        gap = np.full_like(Tk, np.nan)
        heads = np.stack([Tk, base + n * hl * 0.45, base - n * hl * 0.45, Tk, gap], axis=1).reshape(-1, 2)
    else:
        heads = np.full((1, 2), np.nan)
    group = name or color
    traces = [
        go.Scatter(x=shaft[:, 0], y=shaft[:, 1], mode="lines", line=dict(color=color, width=width, dash=dash),
                   name=name, legendgroup=group, showlegend=showlegend and bool(name),
                   hoverinfo="skip", opacity=opacity),
        go.Scatter(x=heads[:, 0], y=heads[:, 1], mode="lines", fill="toself", fillcolor=color,
                   line=dict(color=color, width=1), legendgroup=group, showlegend=False, hoverinfo="skip",
                   opacity=opacity),
    ]
    if labels is not None and len(tips):
        off = np.array([0.18, 0.18])
        traces.append(go.Scatter(x=(tips + off)[:, 0], y=(tips + off)[:, 1], mode="text", text=list(labels),
                                 textfont=dict(color=color, size=15), legendgroup=group, showlegend=False,
                                 hovertext=[f"({a:.2f}, {b:.2f})" for a, b in tips], hoverinfo="text"))
    else:
        hover = [f"{name} ({a:.2f}, {b:.2f})" for a, b in tips] if len(tips) <= 60 else None
        traces.append(go.Scatter(x=tips[:, 0], y=tips[:, 1], mode="markers", marker=dict(size=10, opacity=0),
                                 legendgroup=group, showlegend=False, hovertext=hover,
                                 hoverinfo="text" if hover else "skip"))
    return traces


def grid_2d(M: np.ndarray, extent: float = 6, step: float = 1, color: str = GRID_COLOR,
            width: float = 1.2, name: str = "transformed grid", showlegend: bool = False) -> go.Scatter:
    """Image of the integer grid under the 2×2 matrix ``M`` (one trace)."""
    c = np.arange(-extent, extent + 1e-9, step)
    e = np.full_like(c, extent)
    starts = np.concatenate([np.stack([c, -e], 1), np.stack([-e, c], 1)]) @ M.T
    ends = np.concatenate([np.stack([c, e], 1), np.stack([e, c], 1)]) @ M.T
    P = segments(starts, ends)
    return go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines", line=dict(color=color, width=width), name=name,
                      showlegend=showlegend, hoverinfo="skip")


def axes_2d(M: np.ndarray, extent: float = 6, color: str = "rgba(255,255,255,0.85)") -> go.Scatter:
    a1, b1 = M @ np.array([-extent, 0]), M @ np.array([extent, 0])
    a2, b2 = M @ np.array([0, -extent]), M @ np.array([0, extent])
    return go.Scatter(x=[a1[0], b1[0], None, a2[0], b2[0]], y=[a1[1], b1[1], None, a2[1], b2[1]],
                      mode="lines", line=dict(color=color, width=2), showlegend=False, hoverinfo="skip")


def polygon_2d(points: np.ndarray, color: str, name: str = "", opacity: float = 0.3,
               showlegend: bool = True) -> go.Scatter:
    P = np.vstack([points, points[:1]])
    return go.Scatter(x=P[:, 0], y=P[:, 1], fill="toself", mode="lines", fillcolor=color, opacity=opacity,
                      line=dict(color=color, width=2), name=name, showlegend=showlegend and bool(name),
                      hoverinfo="skip")


def unit_square(M: np.ndarray) -> np.ndarray:
    return (M @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]], float)).T


def circle_points(n: int = 120, r: float = 1.0) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)])


def curve_2d(points: np.ndarray, color: str, name: str = "", width: float = 2.5, dash: str | None = None,
             showlegend: bool = True) -> go.Scatter:
    return go.Scatter(x=points[0], y=points[1], mode="lines", line=dict(color=color, width=width, dash=dash),
                      name=name, showlegend=showlegend and bool(name), hoverinfo="skip")


def line_through_origin_2d(direction: np.ndarray, span: float, color: str, name: str = "",
                           dash: str = "dash", width: float = 2, showlegend: bool = True) -> go.Scatter:
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0) * span * 3
    return go.Scatter(x=[-d[0], d[0]], y=[-d[1], d[1]], mode="lines",
                      line=dict(color=color, dash=dash, width=width), name=name,
                      showlegend=showlegend and bool(name), hoverinfo="skip")


def points_2d(P: np.ndarray, color: str, name: str = "", size: int = 7, showlegend: bool = True,
              symbol: str = "circle") -> go.Scatter:
    P = np.atleast_2d(P)
    return go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", marker=dict(color=color, size=size, symbol=symbol),
                      name=name, showlegend=showlegend and bool(name))


# ---------------------------------------------------------------------------
# 3D primitives
# ---------------------------------------------------------------------------

def _cone_mesh(tips: np.ndarray, dirs: np.ndarray, length, radius, k: int = 14):
    """Vertices and triangles of one cone per tip (vectorised); empty input gives a dummy point."""
    n = len(tips)
    if n == 0:
        z = np.zeros(1)
        return z, z, z, [0], [0], [0]
    length = np.broadcast_to(np.asarray(length, float), (n,))
    radius = np.broadcast_to(np.asarray(radius, float), (n,))
    u = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    a = np.where(np.abs(u[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    base = tips - u * length[:, None]
    th = 2 * np.pi * np.arange(k) / k
    ring = (base[:, None, :] + radius[:, None, None] *
            (np.cos(th)[None, :, None] * e1[:, None, :] + np.sin(th)[None, :, None] * e2[:, None, :]))
    verts = np.concatenate([tips[:, None, :], base[:, None, :], ring], axis=1)  # (n, k+2, 3)
    off = (np.arange(n) * (k + 2))[:, None]
    j = np.arange(k)
    ra, rb = 2 + j, 2 + (j + 1) % k
    I = np.concatenate([np.broadcast_to(0, (n, k)), np.broadcast_to(1, (n, k))], axis=1) + off
    J = np.concatenate([np.broadcast_to(ra, (n, k)), np.broadcast_to(rb, (n, k))], axis=1) + off
    K = np.concatenate([np.broadcast_to(rb, (n, k)), np.broadcast_to(ra, (n, k))], axis=1) + off
    V = verts.reshape(-1, 3)
    return V[:, 0], V[:, 1], V[:, 2], I.ravel(), J.ravel(), K.ravel()


def arrows_3d(vectors: np.ndarray, origins: np.ndarray | None = None, color: str = PALETTE[0],
              name: str = "", width: float = 7, head: float = 0.3, showlegend: bool = True,
              labels: Sequence[str] | None = None, opacity: float = 1.0) -> list:
    """3D arrows (shafts + mesh cones + tip labels); always returns 3 traces."""
    V = np.asarray(vectors, dtype=float).reshape(-1, 3)
    O = np.zeros_like(V) if origins is None else np.broadcast_to(np.asarray(origins, float), V.shape)
    tips = O + V
    L = np.linalg.norm(V, axis=1)
    ok = np.isfinite(L) & (L > 1e-9)
    Vk, Ok, Tk, Lk = V[ok], O[ok], tips[ok], L[ok]
    hl = np.minimum(head, 0.4 * Lk)
    base = Ok + Vk * (1 - hl / Lk)[:, None] if len(Vk) else Ok
    shaft = segments(Ok, base) if len(Vk) else np.full((1, 3), np.nan)
    mx, my, mz, I, J, K = _cone_mesh(Tk, Vk, hl, hl * 0.38)
    group = name or color
    traces = [
        go.Scatter3d(x=shaft[:, 0], y=shaft[:, 1], z=shaft[:, 2], mode="lines", line=dict(color=color, width=width),
                     name=name, legendgroup=group, showlegend=showlegend and bool(name), hoverinfo="skip",
                     opacity=opacity),
        go.Mesh3d(x=mx, y=my, z=mz, i=I, j=J, k=K, color=color, opacity=opacity if len(Tk) else 0, flatshading=True,
                  legendgroup=group, showlegend=False, hoverinfo="skip", lighting=dict(ambient=0.7, diffuse=0.6)),
    ]
    text = list(labels) if labels is not None else None
    hover = [f"{name} ({a:.2f}, {b:.2f}, {c:.2f})" for a, b, c in tips] if len(tips) <= 60 else None
    traces.append(go.Scatter3d(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2], mode="text" if text else "markers",
                               text=text, textfont=dict(color=color, size=14),
                               marker=dict(size=3, color=color, opacity=0.01),
                               hovertext=hover, hoverinfo="text" if hover else "skip", legendgroup=group,
                               showlegend=False))
    return traces


def grid_3d(M: np.ndarray, extent: float = 3, step: float = 1, color: str = GRID_COLOR,
            width: float = 2, name: str = "transformed grid") -> go.Scatter3d:
    vals = np.arange(-extent, extent + 1e-9, step)
    A, B = [g.ravel() for g in np.meshgrid(vals, vals)]
    lo, hi = np.full_like(A, -extent), np.full_like(A, extent)
    starts = np.concatenate([np.stack([lo, A, B], 1), np.stack([A, lo, B], 1), np.stack([A, B, lo], 1)]) @ M.T
    ends = np.concatenate([np.stack([hi, A, B], 1), np.stack([A, hi, B], 1), np.stack([A, B, hi], 1)]) @ M.T
    P = segments(starts, ends)
    return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=color, width=width), name=name,
                        showlegend=False, hoverinfo="skip")


CUBE_VERTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
CUBE_I = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
CUBE_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
CUBE_K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7]
CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def parallelepiped(M: np.ndarray, color: str = "#ffd43b", name: str = "unit cube image",
                   opacity: float = 0.25) -> list:
    V = (M @ CUBE_VERTS.T).T
    mesh = go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=CUBE_I, j=CUBE_J, k=CUBE_K, color=color,
                     opacity=opacity, name=name, showlegend=bool(name), hoverinfo="skip", flatshading=True)
    a, b = np.array(CUBE_EDGES).T
    E = segments(V[a], V[b])
    edges = go.Scatter3d(x=E[:, 0], y=E[:, 1], z=E[:, 2], mode="lines", line=dict(color=color, width=4),
                         showlegend=False, hoverinfo="skip")
    return [mesh, edges]


def sphere_surface(M: np.ndarray, color: str = "#cc5de8", name: str = "unit sphere image",
                   opacity: float = 0.2, n: int = 30) -> go.Surface:
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    P = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    Q = np.einsum("ij,jkl->ikl", M, P)
    return go.Surface(x=Q[0], y=Q[1], z=Q[2], opacity=opacity, showscale=False, name=name,
                      colorscale=[[0, color], [1, color]], hoverinfo="skip", showlegend=bool(name))


def plane_through_origin(u: np.ndarray, v: np.ndarray, span: float, color: str, name: str = "",
                         opacity: float = 0.3, offset: np.ndarray | None = None) -> go.Surface:
    s = np.linspace(-span * 1.5, span * 1.5, 2)
    S, T = np.meshgrid(s, s)
    o = np.zeros(3) if offset is None else offset
    u = u / (np.linalg.norm(u) or 1)
    v = v / (np.linalg.norm(v) or 1)
    P = o[:, None, None] + u[:, None, None] * S + v[:, None, None] * T
    return go.Surface(x=P[0], y=P[1], z=P[2], opacity=opacity, showscale=False, name=name,
                      colorscale=[[0, color], [1, color]], hoverinfo="skip", showlegend=bool(name))


def plane_from_equation(n: np.ndarray, d: float, span: float, color: str, name: str = "",
                        opacity: float = 0.35) -> go.Surface:
    """Plane n·x = d drawn as a square patch."""
    n = np.asarray(n, float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return go.Surface(x=[[0]], y=[[0]], z=[[0]], opacity=0, showscale=False)
    point = n * d / nn ** 2
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 * nn else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    v = np.cross(n, u)
    return plane_through_origin(u, v, span, color, name, opacity, offset=point)


def line_3d(direction: np.ndarray, span: float, color: str, name: str = "", width: float = 5,
            dash: str | None = "dash", offset: np.ndarray | None = None) -> go.Scatter3d:
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0) * span * 3
    o = np.zeros(3) if offset is None else np.asarray(offset, float)
    return go.Scatter3d(x=[o[0] - d[0], o[0] + d[0]], y=[o[1] - d[1], o[1] + d[1]], z=[o[2] - d[2], o[2] + d[2]],
                        mode="lines", line=dict(color=color, width=width, dash=dash), name=name,
                        showlegend=bool(name), hoverinfo="skip")


def points_3d(P: np.ndarray, color: str, name: str = "", size: int = 4, showlegend: bool = True,
              opacity: float = 1.0) -> go.Scatter3d:
    P = np.atleast_2d(P)
    return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                        marker=dict(color=color, size=size, opacity=opacity), name=name,
                        showlegend=showlegend and bool(name))


def polyline_3d(P: np.ndarray, color: str, name: str = "", width: float = 4, showlegend: bool = True,
                dash: str | None = None) -> go.Scatter3d:
    return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=color, width=width, dash=dash),
                        name=name, showlegend=showlegend and bool(name), hoverinfo="skip")


def axes_3d(span: float) -> go.Scatter3d:
    s = span
    return go.Scatter3d(x=[-s, s, None, 0, 0, None, 0, 0], y=[0, 0, None, -s, s, None, 0, 0],
                        z=[0, 0, None, 0, 0, None, -s, s], mode="lines",
                        line=dict(color="rgba(255,255,255,0.5)", width=3), showlegend=False, hoverinfo="skip")


# ---------------------------------------------------------------------------
# Many-arrow fields
# ---------------------------------------------------------------------------

def quiver_2d(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray, scale: float,
              colorscale: str = "Turbo", normalise: bool = True, name: str = "field",
              cmax: float | None = None) -> list:
    """Arrow field coloured by magnitude.

    Arrows are split into colour buckets (one trace each) because a Plotly line
    trace has a single colour.  Always returns ``n_buckets`` traces + 1 colourbar.
    """
    import plotly.colors as pc

    X, Y, U, V = (np.ravel(a) for a in (X, Y, U, V))
    mag = np.hypot(U, V)
    ok = np.isfinite(mag)
    top = cmax if cmax is not None else (np.nanmax(mag[ok]) if ok.any() else 1.0)
    top = top if top > 1e-12 else 1.0
    with np.errstate(all="ignore"):
        if normalise:
            L = np.where(mag > 1e-12, scale / mag, 0) * (0.35 + 0.65 * np.clip(mag / top, 0, 1))
        else:
            L = np.full_like(mag, scale / top)
    du, dv = U * L, V * L
    buckets = 8
    cols = pc.sample_colorscale(colorscale, [(b + 0.5) / buckets for b in range(buckets)])
    idx = np.clip(np.nan_to_num(mag / top * buckets).astype(int), 0, buckets - 1)
    traces = []
    Ln = np.hypot(du, dv)
    for b in range(buckets):
        sel = ok & (idx == b) & (Ln > 1e-12)
        x, y, u, v, ln = X[sel], Y[sel], du[sel], dv[sel], Ln[sel]
        if len(x):
            ux, uy = u / ln, v / ln
            hl = 0.35 * ln
            tx, ty = x + u, y + v
            bx, by = tx - ux * hl, ty - uy * hl
            gap = np.full_like(x, np.nan)
            xs = np.stack([x, tx, gap, bx - uy * hl * 0.5, tx, bx + uy * hl * 0.5, gap], 1).ravel()
            ys = np.stack([y, ty, gap, by + ux * hl * 0.5, ty, by - ux * hl * 0.5, gap], 1).ravel()
        else:
            xs = ys = np.array([np.nan])
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cols[b], width=1.8),
                                 showlegend=False, hoverinfo="skip", name=name))
    traces.append(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=colorscale, cmin=0, cmax=top, color=[0], showscale=True,
                                         colorbar=dict(title="|F|", thickness=12, len=0.6))))
    return traces


def cone_field_3d(X, Y, Z, U, V, W, sizeref: float = 0.6, colorscale: str = "Turbo",
                  cmax: float | None = None, name: str = "field") -> go.Cone:
    mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    top = cmax if cmax is not None else float(np.nanmax(mag)) if np.isfinite(mag).any() else 1
    return go.Cone(x=np.ravel(X), y=np.ravel(Y), z=np.ravel(Z), u=np.ravel(U), v=np.ravel(V), w=np.ravel(W),
                   sizemode="scaled", sizeref=sizeref, anchor="center", colorscale=colorscale, cmin=0,
                   cmax=top or 1, colorbar=dict(title="|F|", thickness=12, len=0.6), name=name,
                   hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>F = (%{u:.2f}, %{v:.2f}, %{w:.2f})<extra></extra>")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animated_figure(make_traces: Callable[[float], list], ts: Iterable[float], layout: go.Layout,
                    static_traces: Sequence = (), duration: int = 60, label: Callable[[float], str] | None = None,
                    initial: int = -1, redraw: bool = False, slider_prefix: str = "t = ") -> go.Figure:
    """Build a figure whose first traces are animated over ``ts``.

    ``make_traces(t)`` must return the same number of traces for every t;
    ``static_traces`` are drawn underneath, unchanged across frames.
    """
    ts = list(ts)
    label = label or (lambda t: f"{t:.2f}")
    static_traces = list(static_traces)
    n_static = len(static_traces)
    first = make_traces(ts[initial])
    fig = go.Figure(data=static_traces + list(first), layout=layout)
    animated_idx = list(range(n_static, n_static + len(first)))
    fig.frames = [go.Frame(data=make_traces(t), traces=animated_idx, name=str(k)) for k, t in enumerate(ts)]
    frame_args = lambda d: dict(frame=dict(duration=d, redraw=redraw), mode="immediate",  # noqa: E731
                                transition=dict(duration=0), fromcurrent=True)
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=1.0, y=1.0, xanchor="right", yanchor="bottom",
                          pad=dict(r=8, t=8), showactive=False, bgcolor="#262730", font=dict(color="#fafafa"),
                          buttons=[dict(label="▶ Play", method="animate", args=[None, frame_args(duration)]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=redraw),
                                                           mode="immediate", transition=dict(duration=0))]),
                                   dict(label="⟲ Start", method="animate", args=[["0"], frame_args(0)])])],
        sliders=[dict(active=initial % len(ts), x=0.0, len=1.0, y=-0.01, yanchor="top", pad=dict(t=8),
                      currentvalue=dict(prefix=slider_prefix, font=dict(size=13, color="#fafafa")),
                      font=dict(color="rgba(0,0,0,0)"), ticklen=0,
                      steps=[dict(method="animate", label=label(t), args=[[str(k)], frame_args(0)])
                             for k, t in enumerate(ts)])],
    )
    fig.update_layout(margin=dict(b=60, t=max(46, fig.layout.margin.t or 0)))
    return fig
