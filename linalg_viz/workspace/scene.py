"""Builds the single workspace graph from the control-panel settings.

Every feature is a *layer*: a function ``ctx -> traces`` plus the things it depends on
("M" = the current transformation matrix, "t" = time).  Layers that depend on what
▶ Play animates are re-drawn in every animation frame; all others are drawn once.
Legends and colour keys are collected in ``Scene.keys`` and shown *outside* the plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import plotly.graph_objects as go

from ..core import linalg as la
from ..core.fields import field_rhs, find_fixed_points, rk4_paths, streamlines
from ..core.parsing import format_number as fmt
from ..viz import plot as P
from .controls import Settings

WIDE = 1.7  # how much wider than tall a 2D view usually is

TURBO_CSS = "linear-gradient(90deg,#30123b,#4686fb,#1ae4b6,#a2fc3c,#fabe2e,#e4460a,#7a0403)"
RDBU_CSS = "linear-gradient(90deg,#053061,#4393c3,#f7f7f7,#d6604d,#67001f)"
PIYG_CSS = "linear-gradient(90deg,#8e0152,#de77ae,#f7f7f7,#7fbc41,#276419)"
VIRIDIS_CSS = "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)"
REDS_CSS = "linear-gradient(90deg,#fee0d2,#fc9272,#de2d26)"
PLASMA_CSS = "linear-gradient(90deg,#0d0887,#9c179e,#ed7953,#f0f921)"


@dataclass
class Ctx:
    M: np.ndarray   # current transformation
    t: float        # current time
    frac: float     # progress of the time range, 0..1


@dataclass
class Key:
    color: str
    label: str
    kind: str = "line"   # line | dash | area | dot | gradient
    detail: str = ""


@dataclass
class Scene:
    s: Settings
    layers: list[tuple[set, Callable[[Ctx], list]]] = field(default_factory=list)
    keys: list[Key] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def dim(self) -> int:
        return self.s.dim

    @property
    def head(self) -> float:
        return self.s.span * (0.045 if self.dim == 2 else 0.07)

    def key(self, color: str, label: str, kind: str = "line", detail: str = "") -> None:
        self.keys.append(Key(color, label, kind, detail))

    def add(self, deps, fn: Callable[[Ctx], list]) -> None:
        self.layers.append((set(deps), fn))

    def arrows(self, V, O=None, color="#fff", width=None, labels=None, opacity=1.0, head=None):
        V = np.asarray(V, float).reshape(-1, self.dim)
        lab = labels if (self.s.labels and labels is not None) else None
        if self.dim == 2:
            return P.arrows_2d(V, O, color=color, width=width or 3.5, head=head or self.head, labels=lab,
                               opacity=opacity, showlegend=False)
        return P.arrows_3d(V, O, color=color, width=width or 8, head=head or self.head, labels=lab,
                           opacity=opacity, showlegend=False)

    def line(self, d, color, dash="dash", width=2.5, offset=None):
        d = np.asarray(d, float)
        L = self.s.span * 4
        d = d / (np.linalg.norm(d) or 1.0) * L
        o = np.zeros(self.dim) if offset is None else np.asarray(offset, float)
        pts = np.stack([o - d, o + d])
        if self.dim == 2:
            return go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, dash=dash, width=width),
                              hoverinfo="skip")
        return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                            line=dict(color=color, dash=dash, width=width * 2), hoverinfo="skip")

    def segment(self, a, b, color, dash="dot", width=2):
        pts = np.stack([a, b])
        if self.dim == 2:
            return go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, dash=dash, width=width),
                              hoverinfo="skip")
        return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                            line=dict(color=color, dash=dash, width=width * 2), hoverinfo="skip")

    def points(self, Pts, color, size=8, symbol="circle", text=None):
        Pts = np.asarray(Pts, float).reshape(-1, self.dim)
        if self.dim == 2:
            return go.Scatter(x=Pts[:, 0], y=Pts[:, 1], mode="markers", marker=dict(color=color, size=size, symbol=symbol),
                              hovertext=text, hoverinfo="text" if text else "x+y")
        return go.Scatter3d(x=Pts[:, 0], y=Pts[:, 1], z=Pts[:, 2], mode="markers",
                            marker=dict(color=color, size=size * 0.6), hovertext=text,
                            hoverinfo="text" if text else "x+y+z")

    def polyline(self, Pts, color, width=2, dash=None, opacity=1.0):
        Pts = np.asarray(Pts, float).reshape(-1, self.dim)
        if len(Pts) == 0:
            Pts = np.full((1, self.dim), np.nan)
        if self.dim == 2:
            return go.Scatter(x=Pts[:, 0], y=Pts[:, 1], mode="lines", line=dict(color=color, width=width, dash=dash),
                              opacity=opacity, hoverinfo="skip")
        return go.Scatter3d(x=Pts[:, 0], y=Pts[:, 1], z=Pts[:, 2], mode="lines",
                            line=dict(color=color, width=width * 2, dash=dash), opacity=opacity, hoverinfo="skip")

    def flat(self, Q, color, opacity=0.25):
        """A subspace with orthonormal basis columns Q, always as [Surface, Scatter3d] in 3D (stable for frames)."""
        span = self.s.span
        d = Q.shape[1]
        if self.dim == 2:
            if d == 1:
                return [self.line(Q[:, 0], color, "solid", 3)]
            if d == 2:
                s_ = span * 1.2
                return [P.polygon_2d(np.array([[-s_, -s_], [s_, -s_], [s_, s_], [-s_, s_]]), color, "", 0.1, False)]
            return [self.points(np.zeros(2), color, 12)]
        surf_op = opacity if d == 2 else 0.0
        u = Q[:, 0] if d >= 1 else np.array([1.0, 0, 0])
        v = Q[:, 1] if d >= 2 else np.array([0, 1.0, 0])
        surf = P.plane_through_origin(u, v, span, color, "", surf_op)
        surf.showlegend = False
        line = self.line(Q[:, 0], color, "solid", 4) if d == 1 else self.polyline(np.zeros((0, 3)), color)
        return [surf, line]

    # ------------------------------------------------------------------ the transformation
    def chain(self):
        steps = self.s.steps if (self.s.tf_on and self.s.steps) else []
        paths = [la.transform_path(st.matrix) for st in steps]

        def M(sv: float) -> np.ndarray:
            out = np.eye(self.dim)
            for k, path in enumerate(paths):
                f = float(np.clip(sv - k, 0, 1))
                if f <= 0:
                    break
                out = path(f) @ out
            return out

        return steps, M

    # ------------------------------------------------------------------ layers
    def base_layer(self) -> None:
        s = self.s
        if self.dim == 2:
            if s.grid:
                extent = max(6, int(np.ceil(s.span * 2)))
                self.add([], lambda c: [P.grid_2d(np.eye(2), extent, color=P.GRID_FAINT, width=1)])
                self.key(P.GRID_FAINT.replace("0.12", "0.5"), "original grid")
        else:
            self.add([], lambda c: [P.axes_3d(s.span)])

    def transform_layer(self) -> None:
        s = self.s
        dim = self.dim
        if s.grid:
            self.key(P.GRID_COLOR, "grid after the matrix")
        if s.shape:
            self.key("#ffd43b", "unit square → area |det|" if dim == 2 else "unit cube → volume |det|", "area")
        if s.circle:
            self.key("#cc5de8", "unit circle → ellipse" if dim == 2 else "unit sphere → ellipsoid",
                     "line" if dim == 2 else "area")
        if s.basis_arrows:
            for k, lab in enumerate("îĵk̂"[:dim] if dim < 3 else ["î", "ĵ", "k̂"]):
                self.key(P.BASIS_COLORS[k], f"{lab} = column {k + 1}")
        if s.fan:
            self.key("rgba(220,220,220,0.8)", "unit vectors (fan)")

        def fn(c: Ctx) -> list:
            M = c.M
            tr = []
            if dim == 2:
                extent = max(6, int(np.ceil(s.span * 2)))
                if s.grid:
                    tr += [P.grid_2d(M, extent), P.axes_2d(M, extent)]
                if s.shape:
                    tr.append(P.polygon_2d(P.unit_square(M), "#ffd43b", "", 0.25, False))
                if s.circle:
                    tr.append(P.curve_2d(M @ P.circle_points(), "#cc5de8", "", 2.5, showlegend=False))
                if s.fan:
                    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
                    tr += self.arrows((M @ np.stack([np.cos(th), np.sin(th)])).T, color="rgba(220,220,220,0.75)",
                                      width=1.5, head=self.head * 0.6)
            else:
                if s.grid:
                    tr.append(P.grid_3d(M, 2, 1, color="rgba(77,171,247,0.2)", width=1.5))
                    ax = P.segments(-3 * M.T, 3 * M.T)
                    tr.append(go.Scatter3d(x=ax[:, 0], y=ax[:, 1], z=ax[:, 2], mode="lines",
                                           line=dict(color="rgba(255,255,255,0.7)", width=3), hoverinfo="skip"))
                if s.shape:
                    tr += P.parallelepiped(M, "#ffd43b", "")
                if s.circle:
                    tr.append(P.sphere_surface(M, name=""))
            if s.basis_arrows:
                labs = ["î", "ĵ", "k̂"]
                for k in range(dim):
                    tr += self.arrows(M[:, k], color=P.BASIS_COLORS[k], labels=[labs[k]],
                                      width=4 if dim == 2 else 10)
            return tr

        self.add(["M"], fn)

    def vector_layer(self) -> None:
        s = self.s
        vecs = s.vectors
        if not vecs:
            return
        dim = self.dim
        deps = ["M"] if s.ride else []
        X = np.column_stack([nv.v for nv in vecs])
        for nv in vecs:
            self.key(nv.color, f"{nv.name} = {fmt_vec(nv.v)}")
        ex = s.vec_extras

        def fn(c: Ctx) -> list:
            M = c.M if s.ride else np.eye(dim)
            W = M @ X  # current vectors as columns
            tr = []
            for k, nv in enumerate(vecs):
                tr += self.arrows(W[:, k], color=nv.color, labels=[nv.name], width=4 if dim == 2 else 10)
            if "sum" in ex:
                origins = np.cumsum(np.column_stack([np.zeros(dim), W[:, :-1]]), axis=1).T
                tr += self.arrows(W.T, origins, color="rgba(255,255,255,0.45)", width=2)
                tr += self.arrows(W.sum(axis=1), color="#ffffff", labels=["sum"], width=5)
            if "linear combination" in ex and s.coeffs:
                cs = np.array(s.coeffs + [0.0] * (len(vecs) - len(s.coeffs)))[: len(vecs)]
                scaled = W * cs
                origins = np.cumsum(np.column_stack([np.zeros(dim), scaled[:, :-1]]), axis=1).T
                for k, nv in enumerate(vecs):
                    tr += self.arrows(scaled[:, k], origins[k], color=nv.color, opacity=0.55, width=2.5)
                tr += self.arrows(scaled.sum(axis=1), color="#f8f9fa", labels=["c·v"], width=5)
            if "span" in ex:
                tr += self.flat(la.column_space(W) if la.rank(W) else np.zeros((dim, 0)), "#ffd43b", 0.18)
            if "projection v₁→v₂" in ex and len(vecs) >= 2:
                a, b_ = W[:, 0], W[:, 1]
                pr = la.project_onto_vector(a, b_)
                tr += self.arrows(pr, color="#ffffff", labels=["proj"], width=5)
                tr.append(self.segment(a, pr, "#adb5bd"))
            if "cross product" in ex and dim == 3 and len(vecs) >= 2:
                a, b_ = W[:, 0], W[:, 1]
                tr += self.arrows(np.cross(a, b_), color=P.color(3), labels=["v₁×v₂"])
                quad = np.array([np.zeros(3), a, a + b_, b_])
                tr.append(go.Mesh3d(x=quad[:, 0], y=quad[:, 1], z=quad[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                                    color="#cc5de8", opacity=0.3, hoverinfo="skip"))
            if "Gram–Schmidt" in ex:
                qs = [st.normalised for st in la.gram_schmidt(W) if st.normalised is not None]
                Q = np.array(qs) if qs else np.zeros((0, dim))
                tr += self.arrows(Q, color="#ffffff", labels=[f"q{k + 1}" for k in range(len(Q))], width=3)
            return tr

        self.add(deps, fn)
        if "sum" in ex:
            self.key("#ffffff", "sum of the vectors (tip to tail)")
        if "linear combination" in ex:
            self.key("#f8f9fa", "c₁v₁ + c₂v₂ + … (sliders)")
        if "span" in ex:
            self.key("#ffd43b", "span of the vectors", "area")
        if "projection v₁→v₂" in ex:
            self.key("#ffffff", "projection of v₁ onto v₂ (dotted: the perpendicular part)")
        if "cross product" in ex and dim == 3:
            self.key(P.color(3), "v₁ × v₂ (⊥ to both, length = parallelogram area)")
        if "Gram–Schmidt" in ex:
            self.key("#ffffff", "q₁, q₂, … orthonormal (Gram–Schmidt)")
        if "power method" in ex and s.A is not None:
            seq = [vecs[0].v / (np.linalg.norm(vecs[0].v) or 1)]
            for _ in range(25):
                w = s.A @ seq[-1]
                nrm = np.linalg.norm(w)
                if nrm < 1e-14:
                    break
                seq.append(w / nrm)
            seq = np.array(seq)
            self.notes["power"] = seq
            self.add([], lambda c: [self.polyline(seq, "rgba(255,212,59,0.5)", 1.5, "dot"),
                                    self.points(seq, "#ffd43b", 7, text=[f"A^{k}v₁ (normalised)" for k in range(len(seq))])])
            self.key("#ffd43b", "A^k v₁ rescaled → swings onto the main eigenvector", "dot")
        if "coordinates in basis B" in ex and s.basis is not None:
            Bm = s.basis
            v = vecs[0].v
            coords = la.change_of_basis(v, Bm)

            def comp(c: Ctx) -> list:
                tr, o = [], np.zeros(dim)
                for k in range(dim):
                    tr += self.arrows(coords[k] * Bm[:, k], o, color=P.BASIS_COLORS[k], opacity=0.6, width=2.5)
                    o = o + coords[k] * Bm[:, k]
                return tr

            self.add([], comp)
            self.key("#adb5bd", f"{vecs[0].name} built from B's vectors: coordinates {fmt_vec(coords)}")

    def eigen_layer(self) -> None:
        s = self.s
        if s.A is None or not s.eig_extras:
            return
        A = s.A
        ex = s.eig_extras
        tr = []
        if "eigenvector lines" in ex:
            cols = [P.EIGEN_COLOR, P.color(3), P.color(4)]
            real = [p for p in la.eigen_analysis(A) if p.is_real]
            for k, p in enumerate(real):
                tr.append(self.line(np.real(p.vector), cols[k % 3], "dash", 2.5))
                self.key(cols[k % 3], f"eigenvector line, λ = {fmt(p.value.real)} (A only stretches it)", "dash")
            if not real:
                self.notes["no_real_eig"] = True
        spaces = {"column space": (la.column_space(A), "#51cf66"), "null space": (la.null_space(A), "#ff6b6b"),
                  "row space": (la.row_space(A), "#4dabf7"), "left null space": (la.left_null_space(A), "#fab005")}
        for name, (Q, color) in spaces.items():
            if name in ex:
                tr += self.flat(Q, color, 0.25)
                what = {0: "just 0", 1: "a line", 2: "a plane" if self.dim == 3 else "everything", 3: "everything"}[Q.shape[1]]
                self.key(color, f"{name} of A ({what})", "area" if Q.shape[1] >= 2 else "line")
        self.add([], lambda c: tr)

    def system_layer(self) -> None:
        s = self.s
        if not s.sys_on or s.A is None or s.b is None:
            return
        A, b = s.A, s.b
        sol = la.solve_system(A, b)
        self.notes["system"] = sol
        tr = []
        names = "xyz"
        if "row picture" in s.sys_extras:
            for i in range(self.dim):
                color = P.color(i + 4)
                eq = " + ".join(f"{fmt(A[i, j])}{names[j]}" for j in range(self.dim)) + f" = {fmt(b[i])}"
                if self.dim == 2:
                    a1, a2 = A[i]
                    span = self.s.span * 3
                    ss = np.linspace(-span, span, 2)
                    if abs(a2) >= abs(a1) and abs(a2) > 1e-12:
                        pts = np.stack([ss, (b[i] - a1 * ss) / a2], 1)
                    elif abs(a1) > 1e-12:
                        pts = np.stack([(b[i] - a2 * ss) / a1, ss], 1)
                    else:
                        continue
                    tr.append(self.polyline(pts, color, 2.5))
                else:
                    surf = P.plane_from_equation(A[i], b[i], self.s.span, color, "", 0.35)
                    tr.append(surf)
                self.key(color, f"equation {i + 1}: {eq}", "line" if self.dim == 2 else "area")
        if "column picture" in s.sys_extras:
            x = sol.particular if sol.particular is not None else sol.least_squares
            o = np.zeros(self.dim)
            for j in range(self.dim):
                tr += self.arrows(x[j] * A[:, j], o, color=P.BASIS_COLORS[j], opacity=0.7, width=3)
                o = o + x[j] * A[:, j]
            tr += self.arrows(b, color="#ffffff", labels=["b"], width=4)
            self.key("#ffffff", "b, reached by x₁·(col 1) + x₂·(col 2) + …")
        if "solution" in s.sys_extras:
            if sol.kind == "unique":
                tr.append(self.points(sol.particular, "#ffffff", 14, "x", [f"solution {fmt_vec(sol.particular)}"]))
                self.key("#ffffff", f"solution x = {fmt_vec(sol.particular)}", "dot")
            elif sol.kind == "infinite":
                N = sol.null_basis
                if N.shape[1] == 1:
                    tr.append(self.line(N[:, 0], "#ffffff", "solid", 4, offset=sol.particular))
                elif self.dim == 3 and N.shape[1] == 2:
                    tr.append(P.plane_through_origin(N[:, 0], N[:, 1], self.s.span, "#ffffff", "", 0.3,
                                                     offset=sol.particular))
                self.key("#ffffff", "all solutions (infinitely many)")
            else:
                tr.append(self.points(sol.least_squares, "#ffffff", 14, "x", ["closest (least squares)"]))
                self.key("#ffffff", f"no solution — best fit x̂ = {fmt_vec(sol.least_squares)}", "dot")
        self.add([], lambda c: tr)

    def basis_layer(self) -> None:
        s = self.s
        if not s.basis_on or s.basis is None:
            return
        Bm = s.basis

        def fn(c: Ctx) -> list:
            if self.dim == 2:
                tr = [P.grid_2d(Bm, max(6, int(s.span * 2)), color="rgba(255,146,43,0.4)", width=1.2)]
            else:
                tr = [P.grid_3d(Bm, 1, 1, color="rgba(255,146,43,0.35)", width=2)]
            for k in range(self.dim):
                tr += self.arrows(Bm[:, k], color="#ff922b", labels=[f"b{k + 1}"], width=3)
            return tr

        self.add([], fn)
        self.key("#ff922b", "basis B and its grid (orange)")

    # ------------------------------------------------------------------ fields
    def _field_cmax(self, F) -> float:
        n = 12 if self.dim == 2 else 6
        ax = np.linspace(-self.s.span, self.s.span, n)
        G = np.meshgrid(*([ax] * self.dim))
        top = 0.0
        for t in ({self.s.t0, (self.s.t0 + self.s.t1) / 2, self.s.t1} if F.time_dependent else {0.0}):
            m = np.linalg.norm(F(*G, t=t), axis=0)
            if np.isfinite(m).any():
                top = max(top, float(np.nanpercentile(m, 99)))
        return top or 1.0

    def field_layer(self) -> None:
        s = self.s
        F = s.F
        if not s.field_on or F is None:
            return
        dim = self.dim
        span = s.span
        deps = ["t"] if F.time_dependent else []
        cmax = self._field_cmax(F)
        self.notes["field_cmax"] = cmax
        n = s.density
        ax = np.linspace(-span, span, n)
        wide = WIDE * span  # 2D views are wider than tall: cover the whole visible width
        axx = np.linspace(-wide, wide, int(round(n * WIDE))) if dim == 2 else ax
        if s.arrows:
            self.key("", "arrow colour = strength |F|", "gradient", f"{TURBO_CSS}|0|{fmt(cmax)}")
        if s.stream:
            self.key("rgba(255,255,255,0.6)", "streamlines (follow the arrows at this instant)")
        bg_range = None
        if dim == 2 and s.background != "none":
            Xb, Yb = np.meshgrid(np.linspace(-wide, wide, 120), np.linspace(-span, span, 80))
            vals = self._background(F, s.background, Xb, Yb, s.t0)
            hi = float(np.nanpercentile(np.abs(vals), 98)) or 1.0
            bg_range = (0, hi) if s.background.startswith("strength") else (-hi, hi)
            css = {"strength": VIRIDIS_CSS, "divergence": RDBU_CSS, "curl": PIYG_CSS}[s.background.split()[0]]
            self.key("", f"background: {s.background}", "gradient", f"{css}|{fmt(bg_range[0])}|{fmt(bg_range[1])}")

        def fn(c: Ctx) -> list:
            tr = []
            if dim == 2:
                if bg_range is not None:
                    xs, ys = np.linspace(-wide, wide, 120), np.linspace(-span, span, 80)
                    Xb, Yb = np.meshgrid(xs, ys)
                    Z = self._background(F, s.background, Xb, Yb, c.t)
                    cs = {"strength": "Viridis", "divergence": "RdBu_r", "curl": "PiYG"}[s.background.split()[0]]
                    tr.append(go.Heatmap(x=xs, y=ys, z=Z, colorscale=cs, zmin=bg_range[0], zmax=bg_range[1],
                                         opacity=0.55, showscale=False,
                                         hovertemplate="value %{z:.3f}<extra></extra>"))
                if s.arrows:
                    X, Y = np.meshgrid(axx, ax)
                    U, V = F(X, Y, t=c.t)
                    tr += P.quiver_2d(X, Y, U, V, 2 * span / n * 0.85, normalise=s.normalise, cmax=cmax)[:-1]
                if s.stream:
                    g = np.linspace(-span, span, 9)
                    seeds = np.stack([m.ravel() for m in np.meshgrid(np.linspace(-wide, wide, 14), g)], 1)
                    lines = streamlines(F, seeds, c.t, length=span, steps=50)
                    S = P.nan_join([L[(np.abs(L[:, 0]) <= wide * 1.05) & (np.abs(L[:, 1]) <= span * 1.05)]
                                    for L in lines])
                    tr.append(go.Scatter(x=S[:, 0], y=S[:, 1], mode="lines", hoverinfo="skip",
                                         line=dict(color="rgba(255,255,255,0.55)", width=1.2)))
            else:
                if s.arrows:
                    X, Y, Z = np.meshgrid(ax, ax, ax)
                    U, V, W = F(X, Y, Z, t=c.t)
                    if s.normalise:
                        m = np.sqrt(U ** 2 + V ** 2 + W ** 2)
                        with np.errstate(all="ignore"):
                            k = np.where(m > 1e-12, (0.35 + 0.65 * np.clip(m / cmax, 0, 1)) / m, 0)
                        U, V, W = U * k, V * k, W * k
                    cone = P.cone_field_3d(X, Y, Z, U, V, W, sizeref=0.32, cmax=1 if s.normalise else cmax)
                    cone.showscale = False
                    tr.append(cone)
                if s.stream:
                    rng = np.random.default_rng(1)
                    seeds = rng.uniform(-span, span, (25, 3))
                    lines = streamlines(F, seeds, c.t, length=span * 1.6, steps=80)
                    S = P.nan_join([L[np.all(np.abs(L) <= span * 1.05, axis=1)] for L in lines], 3)
                    tr.append(go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="lines", hoverinfo="skip",
                                           line=dict(color="rgba(255,255,255,0.6)", width=2)))
            return tr

        self.add(deps, fn)

    @staticmethod
    def _background(F, kind, X, Y, t):
        if kind.startswith("strength"):
            return np.linalg.norm(F(X, Y, t=t), axis=0)
        if kind.startswith("divergence"):
            return F.divergence_grid([X, Y], t)
        return F.curl_grid([X, Y], t)

    def scalar_layer(self) -> None:
        s = self.s
        f = s.f
        if not s.scalar_on or f is None:
            return
        span = s.span
        deps = ["t"] if "t" in f.expr.uses else []
        xs = np.linspace(-span, span, 70 if self.dim == 2 else 22)
        xw = np.linspace(-WIDE * span, WIDE * span, 110) if self.dim == 2 else xs
        if self.dim == 2:
            X, Y = np.meshgrid(xw, xs)
            Z0 = f(X, Y, t=s.t0)
        else:
            X, Y, Z = np.meshgrid(xs, xs, xs)
            Z0 = f(X, Y, Z, t=s.t0)
        lo, hi = float(np.nanpercentile(Z0, 2)), float(np.nanpercentile(Z0, 98))
        if hi - lo < 1e-12:
            hi = lo + 1
        if s.contours:
            self.key("", "level sets of f (colour = value)", "gradient", f"{VIRIDIS_CSS}|{fmt(lo)}|{fmt(hi)}")
        if s.grad_arrows:
            self.key("", "∇f arrows (point uphill, ⊥ level sets)", "gradient", f"{REDS_CSS}|small|steep")
        n = 15 if self.dim == 2 else 6
        g = np.linspace(-span, span, n)

        def fn(c: Ctx) -> list:
            tr = []
            if self.dim == 2:
                if s.contours:
                    tr.append(go.Contour(x=xw, y=xs, z=f(X, Y, t=c.t), colorscale="Viridis", zmin=lo, zmax=hi,
                                         contours=dict(start=lo, end=hi, size=(hi - lo) / 14, coloring="lines"),
                                         line=dict(width=2), showscale=False,
                                         hovertemplate="f = %{z:.3f}<extra></extra>"))
                if s.grad_arrows:
                    Xg, Yg = np.meshgrid(np.linspace(-WIDE * span, WIDE * span, int(n * WIDE)), g)
                    G = f.gradient_grid([Xg, Yg], c.t)
                    tr += P.quiver_2d(Xg, Yg, G[0], G[1], 2 * span / n * 0.8, colorscale="Reds")[:-1]
            else:
                if s.contours:
                    tr.append(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=f(X, Y, Z, t=c.t).ravel(),
                                            isomin=lo, isomax=hi, surface_count=4, opacity=0.25, colorscale="Viridis",
                                            showscale=False, caps=dict(x_show=False, y_show=False, z_show=False)))
                if s.grad_arrows:
                    Xg, Yg, Zg = np.meshgrid(g, g, g)
                    G = f.gradient_grid([Xg, Yg, Zg], c.t)
                    cone = P.cone_field_3d(Xg, Yg, Zg, *G, sizeref=0.35, colorscale="Reds")
                    cone.showscale = False
                    tr.append(cone)
            return tr

        self.add(deps, fn)

    def particle_layer(self) -> None:
        s = self.s
        F = s.F
        if not s.particles_on or F is None:
            return
        dim = self.dim
        rng = np.random.default_rng(7)
        c, r, n = s.centre, s.radius, s.count
        n_ring = 0
        if s.seeding.startswith("blob"):
            if dim == 2:
                th = np.linspace(0, 2 * np.pi, 100, endpoint=False)
                ring = c + r * np.stack([np.cos(th), np.sin(th)], 1)
                rr = r * np.sqrt(rng.uniform(0, 1, n))
                aa = rng.uniform(0, 2 * np.pi, n)
                seeds = np.vstack([ring, c + np.stack([rr * np.cos(aa), rr * np.sin(aa)], 1)])
                n_ring = len(ring)
            else:
                v = rng.normal(size=(n, 3))
                seeds = c + r * v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(0, 1, (n, 1)) ** (1 / 3)
        elif s.seeding == "grid":
            k = max(2, int(round(n ** (1 / dim))))
            g = np.linspace(-s.span * 0.9, s.span * 0.9, k)
            seeds = np.stack([m.ravel() for m in np.meshgrid(*([g] * dim))], 1)
        elif s.seeding == "random":
            hi = np.array([WIDE * s.span, s.span, s.span][:dim])
            seeds = rng.uniform(-hi, hi, (n, dim))
        else:
            d = np.zeros(dim)
            d[0] = r
            seeds = c + np.linspace(-1, 1, n)[:, None] * d
        steps = max(400, s.frames * 8)
        times, paths = rk4_paths(field_rhs(F), seeds, s.t0, s.t1, steps, bound=s.span * 60)
        self.notes["particles"] = (times, paths, n_ring)
        colors = np.linspace(0, 1, len(seeds) - n_ring)
        self.key("", "particles (colour = which particle)", "gradient", f"{PLASMA_CSS}||")
        if s.trail:
            self.key("rgba(77,171,247,0.6)", "particle trails")
        if n_ring:
            self.key("#ffd43b", "blob of fluid (its area changes by the divergence)", "area")

        def fn(cx: Ctx) -> list:
            i = int(np.clip(np.searchsorted(times, cx.t), 0, len(times) - 1))
            Pk = paths[i]
            j0 = max(0, i - int(s.trail * steps / max(s.frames, 1)))
            seg = paths[j0: i + 1: max(1, (i - j0) // 30 or 1)]
            tr = []
            if dim == 2:
                ring = Pk[:n_ring] if n_ring else np.full((1, 2), np.nan)
                ring = np.vstack([ring, ring[:1]])
                tr.append(go.Scatter(x=ring[:, 0], y=ring[:, 1], fill="toself", fillcolor="rgba(255,212,59,0.22)",
                                     mode="lines", line=dict(color="#ffd43b", width=2), hoverinfo="skip"))
                T = P.nan_join([seg[:, k] for k in range(n_ring, len(seeds))] if s.trail and len(seg) > 1 else [])
                tr.append(go.Scatter(x=T[:, 0], y=T[:, 1], mode="lines", hoverinfo="skip",
                                     line=dict(color="rgba(77,171,247,0.5)", width=1.2)))
                tr.append(go.Scatter(x=Pk[n_ring:, 0], y=Pk[n_ring:, 1], mode="markers", hoverinfo="skip",
                                     marker=dict(size=6, color=colors, colorscale="Plasma")))
            else:
                T = P.nan_join([seg[:, k] for k in range(len(seeds))] if s.trail and len(seg) > 1 else [], 3)
                tr.append(go.Scatter3d(x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines", hoverinfo="skip",
                                       line=dict(color="rgba(77,171,247,0.6)", width=2)))
                tr.append(go.Scatter3d(x=Pk[:, 0], y=Pk[:, 1], z=Pk[:, 2], mode="markers", hoverinfo="skip",
                                       marker=dict(size=3, color=colors, colorscale="Plasma")))
            return tr

        self.add(["t"], fn)
        if s.fixed_points and not F.time_dependent:
            fps = find_fixed_points(F, s.window, seeds_per_axis=9 if dim == 2 else 5)
            self.notes["fixed_points"] = fps
            if fps:
                pts = np.array([fp.point for fp in fps])
                self.add([], lambda c: [self.points(pts, "#ff6b6b", 13, "x", [f"{fp.kind} at {fmt_vec(fp.point)}"
                                                                           for fp in fps])])
                self.key("#ff6b6b", "fixed points (F = 0)", "dot")

    def probe_layer(self) -> None:
        s = self.s
        if not s.probe_on or s.p is None:
            return
        dim = self.dim
        p, r = s.p, s.probe_r
        F = s.F
        ex = s.probe_extras
        self.key("#ffffff", f"probe at {fmt_vec(p)}", "dot")
        self.add([], lambda c: [self.points(p, "#ffffff", 11)])
        if F is not None and "flux (divergence)" in ex:
            self.key("#51cf66", "field flowing out through the probe circle", "line")
            self.key("#ff6b6b", "field flowing in", "line")
            if dim == 2:
                th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                nrm = np.stack([np.cos(th), np.sin(th)], 1)
            else:
                k = np.arange(26) + 0.5
                ph, tt = np.arccos(1 - 2 * k / 26), np.pi * (1 + 5 ** 0.5) * k
                nrm = np.stack([np.cos(tt) * np.sin(ph), np.sin(tt) * np.sin(ph), np.cos(ph)], 1)
            pts = p + r * nrm

            def flux(c: Ctx) -> list:
                Fv = F(*pts.T, t=c.t).T
                fn_ = np.sum(Fv * nrm, axis=1)
                L = r * 0.9 / (np.max(np.abs(fn_)) or 1)
                out = fn_ >= 0
                circ = (p[:, None] + r * P.circle_points(100)).T if dim == 2 else np.full((1, 3), np.nan)
                return ([self.polyline(circ, "rgba(255,255,255,0.8)", 1.5)]
                        + self.arrows((nrm * fn_[:, None] * L)[out], pts[out], color="#51cf66", width=2.5,
                                      head=r * 0.25)
                        + self.arrows((nrm * fn_[:, None] * L)[~out], pts[~out], color="#ff6b6b", width=2.5,
                                      head=r * 0.25))

            self.add(["t"] if F.time_dependent else [], flux)
        if F is not None and "paddle wheel (curl)" in ex:
            self.key("#fcc419", "paddle wheel: spins at curl/2 while time runs", "line")
            t0 = s.t0

            def paddle(c: Ctx) -> list:
                curl = F.curl_at(p, c.t)
                if dim == 2:
                    ang = curl / 2 * (c.t - t0) + np.arange(4) * np.pi / 2
                    ends = p + r * np.stack([np.cos(ang), np.sin(ang)], 1)
                else:
                    cn = np.linalg.norm(curl)
                    axis = curl / cn if cn > 1e-9 else np.array([0, 0, 1.0])
                    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
                    e1 = np.cross(axis, a)
                    e1 /= np.linalg.norm(e1)
                    e2 = np.cross(axis, e1)
                    ang = cn / 2 * (c.t - t0) + np.arange(4) * np.pi / 2
                    ends = p + r * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
                seg = P.segments(np.repeat(p[None], 4, 0), ends)
                return [self.polyline(seg, "#fcc419", 4)]

            self.add(["t"], paddle)
        if F is not None and "local linear map (Jacobian)" in ex:
            J = F.jacobian(p, s.t_now)
            self.notes["J"] = J
            tr = []
            for k, pr in enumerate(la.eigen_analysis(J)):
                if pr.is_real:
                    d = np.real(pr.vector)
                    d = d / np.linalg.norm(d) * r * 1.6
                    tr.append(self.segment(p - d, p + d, P.color(4 + k), "solid", 3))
            if dim == 2:
                sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float) * r * 0.6
                moved = sq @ la.matrix_exp(J * 0.5).T
                tr.append(self.polyline(p + sq, "rgba(255,255,255,0.7)", 1.5))
                tr.append(P.polygon_2d(p + moved[:-1], "#22b8cf", "", 0.25, False))
                self.key("#22b8cf", "small square after flowing ½ time unit (≈ e^{J/2})", "area")
            self.key(P.color(4), "eigen-directions of the Jacobian J at the probe")
            self.add([], lambda c: tr)
        if s.f is not None and "gradient of f" in ex:
            g = s.f.gradient_grid([np.array(v) for v in p], s.t_now).ravel()
            gn = np.linalg.norm(g)
            if gn > 1e-12:
                self.add([], lambda c: self.arrows(g / gn * s.span * 0.2, p, color="#ff8787", labels=["∇f"]))
                self.key("#ff8787", f"∇f at the probe (steepest uphill), |∇f| = {fmt(gn)}")

    def curve_layer(self) -> None:
        s = self.s
        cv = s.curve
        if not s.curve_on or cv is None:
            return
        dim = self.dim
        u = np.linspace(s.c0, s.c1, 500)
        R = cv(u)
        self.add([], lambda c: [self.polyline(R, "rgba(77,171,247,0.9)", 3)])
        self.key("rgba(77,171,247,0.9)", "path of r(t)")
        self.key("#ffffff", "moving point r(t)", "dot")
        ex = s.curve_extras
        scale = s.span * 0.15
        if "velocity" in ex:
            self.key(P.color(0), "velocity r′(t)")
        if "acceleration" in ex:
            self.key(P.color(4), "acceleration r″(t)")
        if "T N B frame" in ex and dim == 3:
            self.key(P.I_COLOR, "T tangent · N normal · B binormal (unit)")
        if "osculating circle" in ex and dim == 2:
            self.key("#22b8cf", "osculating circle (best-fitting circle)", "dash")
        vmax = float(np.nanmax(cv.kinematics(u)["speed"])) or 1.0

        def fn(c: Ctx) -> list:
            uu = s.c0 + c.frac * (s.c1 - s.c0)
            K = cv.kinematics(np.array([uu]))
            r_ = K["r"][0]
            tr = [self.points(r_, "#ffffff", 13)]
            vel = K["v"][0] * scale / vmax * 2 if "velocity" in ex else np.zeros((0, dim))
            acc = K["a"][0] * scale / vmax * 2 if "acceleration" in ex else np.zeros((0, dim))
            tr += self.arrows(vel, r_, color=P.color(0))
            tr += self.arrows(acc, r_, color=P.color(4))
            if dim == 3:
                for vec, colr, lab in ((K["T"][0], P.I_COLOR, "T"), (K["N"][0], P.J_COLOR, "N"),
                                       (K["B"][0], P.K_COLOR, "B")):
                    ok = "T N B frame" in ex and np.all(np.isfinite(vec))
                    tr += self.arrows(vec * scale * 1.5 if ok else np.zeros((0, 3)), r_, color=colr,
                                      labels=[lab] if ok else None)
            if dim == 2:
                kap = K["curvature"][0]
                if "osculating circle" in ex and np.isfinite(kap) and kap > 1e-6 and 1 / kap < 1e3:
                    centre = r_ + K["N"][0][:2] / kap
                    tr.append(self.polyline((centre[:, None] + P.circle_points(120, 1 / kap)).T, "#22b8cf", 1.5, "dot"))
                else:
                    tr.append(self.polyline(np.zeros((0, 2)), "#22b8cf"))
            return tr

        self.add(["t"], fn)

    # ------------------------------------------------------------------ build
    def build(self, mode: str) -> go.Figure:
        s = self.s
        dim = self.dim
        steps, chain = self.chain()
        n_steps = len(steps)
        M_final = chain(n_steps)
        self.notes["M_final"] = M_final
        frac_now = (s.t_now - s.t0) / (s.t1 - s.t0)

        self.base_layer()
        self.eigen_layer()
        self.basis_layer()
        self.scalar_layer()
        self.field_layer()
        self.system_layer()
        self.transform_layer()
        self.vector_layer()
        self.particle_layer()
        self.probe_layer()
        self.curve_layer()

        if mode == "time":
            animated = {"t"} | ({"M"} if s.space_flow and s.A is not None else set())
            ts = np.linspace(s.t0, s.t1, s.frames)
            if s.space_flow and s.A is not None:
                Mt = lambda t: la.matrix_exp((t - s.t0) * s.A)  # noqa: E731
            else:
                Mt = lambda t: M_final  # noqa: E731
            ctxs = [Ctx(Mt(t), float(t), k / max(len(ts) - 1, 1)) for k, t in enumerate(ts)]
            labels = [f"t = {t:.2f}" for t in ts]
            static_ctx = Ctx(Mt(s.t_now), s.t_now, frac_now)
            initial = 0
        else:
            animated = {"M"}
            per = 30 if n_steps <= 2 else 24
            ss = np.linspace(0, max(n_steps, 1), per * max(n_steps, 1) + 1)
            ctxs = [Ctx(chain(sv), s.t_now, frac_now) for sv in ss]
            labels = []
            for sv in ss:
                if not steps:
                    labels.append("no matrix applied")
                    continue
                k = min(int(sv), n_steps - 1)
                labels.append(f"step {k + 1}/{n_steps}: {steps[k].name}  {int(round((sv - k) * 100))}%")
            static_ctx = Ctx(M_final, s.t_now, frac_now)
            initial = -1
        static, dynamic = [], []
        for deps, fn in self.layers:
            (dynamic if deps & animated else static).append(fn)
        static_traces = [tr for fn in static for tr in fn(static_ctx)]

        def frame(k: int) -> list:
            return [tr for fn in dynamic for tr in fn(ctxs[int(k)])]

        layout = P.layout_2d(s.span, 820) if dim == 2 else P.layout_3d(s.span, 820)
        layout.showlegend = False
        if not dynamic:
            fig = go.Figure(static_traces, layout)
            return fig
        redraw = dim == 3 or any(not isinstance(t, go.Scatter) for t in frame(initial))
        fig = P.animated_figure(frame, range(len(ctxs)), layout, static_traces, duration=50 if dim == 2 else 80,
                                label=lambda k: labels[int(k)], initial=initial, redraw=redraw, slider_prefix="")
        fig.update_layout(showlegend=False)
        return fig


def fmt_vec(v) -> str:
    return "(" + ", ".join(fmt(x) for x in np.asarray(v).ravel()) + ")"
