"""Reusable animated scenes: space being transformed by a path of matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

from ..core import linalg as la
from . import plot as P


@dataclass
class SceneOptions:
    span: float = 4.0
    grid: bool = True
    grid_extent: float | None = None
    shape: bool = True      # unit square / unit cube
    circle: bool = False    # unit circle / unit sphere
    basis: bool = True
    fan: int = 0            # number of unit vectors around the circle (2D) to show how directions move
    vectors: Sequence[tuple[np.ndarray, str, str]] = field(default_factory=list)  # (v, colour, name)
    lines: Sequence[tuple[np.ndarray, str, str]] = field(default_factory=list)    # static lines through 0
    height: int = 640


def _frame_2d(M: np.ndarray, o: SceneOptions) -> list:
    span = o.span
    head = span * 0.045
    extent = o.grid_extent or max(6, int(np.ceil(span * 1.6)))
    tr: list = []
    if o.grid:
        tr.append(P.grid_2d(M, extent))
        tr.append(P.axes_2d(M, extent))
    if o.shape:
        tr.append(P.polygon_2d(P.unit_square(M), "#ffd43b", "unit square → area |det|", 0.25))
    if o.circle:
        tr.append(P.curve_2d(M @ P.circle_points(), "#cc5de8", "unit circle", 2.5))
    if o.fan:
        th = np.linspace(0, 2 * np.pi, o.fan, endpoint=False)
        U = np.stack([np.cos(th), np.sin(th)])
        tr += P.arrows_2d((M @ U).T, color="rgba(220,220,220,0.75)", name="unit vectors", width=1.6,
                          head=head * 0.6)
    if o.basis:
        tr += P.arrows_2d(M[:, 0], color=P.I_COLOR, name="î (1st column)", head=head, labels=["î"])
        tr += P.arrows_2d(M[:, 1], color=P.J_COLOR, name="ĵ (2nd column)", head=head, labels=["ĵ"])
    for v, c, name in o.vectors:
        tr += P.arrows_2d(M @ v, color=c, name=name, head=head)
    return tr


def _frame_3d(M: np.ndarray, o: SceneOptions) -> list:
    span = o.span
    head = span * 0.07
    extent = o.grid_extent or 2
    tr: list = []
    if o.grid:
        tr.append(P.grid_3d(M, extent, 1, color="rgba(77,171,247,0.18)", width=1.5))
        ax = P.segments(-extent * M.T, extent * M.T)
        tr.append(go.Scatter3d(x=ax[:, 0], y=ax[:, 1], z=ax[:, 2], mode="lines",
                               line=dict(color="rgba(255,255,255,0.7)", width=3), showlegend=False, hoverinfo="skip"))
    if o.shape:
        tr += P.parallelepiped(M, "#ffd43b", "unit cube → volume |det|")
    if o.circle:
        tr.append(P.sphere_surface(M))
    if o.basis:
        for k, (c, lab) in enumerate(zip(P.BASIS_COLORS, ("î", "ĵ", "k̂"))):
            tr += P.arrows_3d(M[:, k], color=c, width=10, name=f"{lab} ({k + 1}st column)" if k == 0 else
                              f"{lab} ({k + 1}{'nd' if k == 1 else 'rd'} column)", head=head, labels=[lab])
    for v, c, name in o.vectors:
        tr += P.arrows_3d(M @ v, color=c, name=name, head=head, width=10)
    return tr


def static_background(dim: int, o: SceneOptions) -> list:
    tr: list = []
    if dim == 2:
        if o.grid:
            extent = o.grid_extent or max(6, int(np.ceil(o.span * 1.6)))
            tr.append(P.grid_2d(np.eye(2), extent, color=P.GRID_FAINT, width=1))
        for d, c, name in o.lines:
            tr.append(P.line_through_origin_2d(d, o.span, c, name))
    else:
        tr.append(P.axes_3d(o.span))
        for d, c, name in o.lines:
            tr.append(P.line_3d(d, o.span, c, name))
    return tr


def transform_figure(dim: int, path: Callable[[float], np.ndarray], ts: Sequence[float], o: SceneOptions,
                     label: Callable[[float], str] | None = None, initial: int = -1,
                     duration: int | None = None, slider_prefix: str = "t = ") -> go.Figure:
    frame = _frame_2d if dim == 2 else _frame_3d
    layout = P.layout_2d(o.span, o.height) if dim == 2 else P.layout_3d(o.span, o.height)
    return P.animated_figure(lambda t: frame(path(t), o), ts, layout, static_background(dim, o),
                             duration=duration or (45 if dim == 2 else 70), label=label, initial=initial,
                             redraw=dim == 3, slider_prefix=slider_prefix)


def linear_path(A: np.ndarray) -> Callable[[float], np.ndarray]:
    return lambda t: la.interpolate_matrix(A, t)


def orthogonal_power(Q: np.ndarray, t: float) -> np.ndarray:
    """Qᵗ for a rotation (det +1) so animations turn instead of shrinking; else linear."""
    n = Q.shape[0]
    if np.linalg.det(Q) > 0 and np.allclose(Q.T @ Q, np.eye(n), atol=1e-8):
        w, V = np.linalg.eig(Q)
        if abs(np.linalg.det(V)) > 1e-8:
            with np.errstate(all="ignore"):
                R = V @ np.diag(w.astype(complex) ** t) @ np.linalg.inv(V)
            if np.allclose(R.imag, 0, atol=1e-7):
                return R.real
    return la.interpolate_matrix(Q, t)


def factor_path(factors: Sequence[np.ndarray], rotation_aware: bool = True) -> Callable[[float], np.ndarray]:
    """Path that applies ``factors`` one after another (first element first).

    For t in [k, k+1] the k-th factor is being applied on top of the earlier ones.
    """

    def path(t: float) -> np.ndarray:
        n = factors[0].shape[0]
        M = np.eye(n)
        for k, F in enumerate(factors):
            s = float(np.clip(t - k, 0, 1))
            if s <= 0:
                break
            ortho = np.allclose(F.T @ F, np.eye(n), atol=1e-8)
            step = orthogonal_power(F, s) if (rotation_aware and ortho) else la.interpolate_matrix(F, s)
            M = step @ M
        return M

    return path
