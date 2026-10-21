"""Vector fields, differential operators and time evolution (flows).

A field is defined by one expression per component in the variables
``x, y`` (2D) or ``x, y, z`` (3D) plus time ``t``.  Every operator is
available numerically (finite differences, works for any expression) and
symbolically through SymPy when the expression allows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .linalg import classify_linear_system_2d
from .parsing import Expression, compile_expr

VARS_2D = ("x", "y", "t")
VARS_3D = ("x", "y", "z", "t")


@dataclass
class VectorField:
    """F(x, y[, z], t) given as component expressions."""

    components: Sequence[str]
    exprs: list[Expression] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.components) not in (2, 3):
            raise ValueError("A vector field needs 2 or 3 components")
        names = self.variables
        self.exprs = [compile_expr(c, names) for c in self.components]

    @property
    def dim(self) -> int:
        return len(self.components)

    @property
    def variables(self) -> tuple[str, ...]:
        return VARS_2D if self.dim == 2 else VARS_3D

    @property
    def time_dependent(self) -> bool:
        return any("t" in e.uses for e in self.exprs)

    def __call__(self, *coords, t: float = 0.0) -> np.ndarray:
        """Evaluate at broadcastable coordinate arrays; returns shape (dim, ...)."""
        coords = [np.asarray(c, dtype=float) for c in coords]
        shape = np.broadcast_shapes(*[c.shape for c in coords])
        tt = np.full(shape, float(t))
        return np.stack([e(*coords, tt) for e in self.exprs])

    def at(self, point: Sequence[float], t: float = 0.0) -> np.ndarray:
        return self(*[np.array(p, dtype=float) for p in point], t=t)

    # ------------------------------------------------------------------
    # Numeric differential operators (central differences)
    # ------------------------------------------------------------------
    def jacobian(self, point: Sequence[float], t: float = 0.0, h: float = 1e-5) -> np.ndarray:
        p = np.asarray(point, dtype=float)
        J = np.zeros((self.dim, self.dim))
        for j in range(self.dim):
            dp = np.zeros(self.dim)
            dp[j] = h
            J[:, j] = (self.at(p + dp, t) - self.at(p - dp, t)) / (2 * h)
        return J

    def divergence_at(self, point, t: float = 0.0) -> float:
        return float(np.trace(self.jacobian(point, t)))

    def curl_at(self, point, t: float = 0.0):
        J = self.jacobian(point, t)
        if self.dim == 2:
            return float(J[1, 0] - J[0, 1])
        return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])

    def divergence_grid(self, grids: Sequence[np.ndarray], t: float = 0.0, h: float = 1e-4) -> np.ndarray:
        total = np.zeros_like(grids[0], dtype=float)
        for i in range(self.dim):
            plus = [g + (h if k == i else 0) for k, g in enumerate(grids)]
            minus = [g - (h if k == i else 0) for k, g in enumerate(grids)]
            total += (self.exprs[i](*plus, t) - self.exprs[i](*minus, t)) / (2 * h)
        return total

    def partial_grid(self, comp: int, var: int, grids, t: float = 0.0, h: float = 1e-4) -> np.ndarray:
        plus = [g + (h if k == var else 0) for k, g in enumerate(grids)]
        minus = [g - (h if k == var else 0) for k, g in enumerate(grids)]
        return (self.exprs[comp](*plus, t) - self.exprs[comp](*minus, t)) / (2 * h)

    def curl_grid(self, grids, t: float = 0.0):
        d = lambda c, v: self.partial_grid(c, v, grids, t)  # noqa: E731
        if self.dim == 2:
            return d(1, 0) - d(0, 1)
        return np.stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])

    # ------------------------------------------------------------------
    # Symbolic forms (LaTeX), when SymPy can handle the expressions
    # ------------------------------------------------------------------
    def symbolic(self):
        exprs = [e.to_sympy() for e in self.exprs]
        if any(e is None for e in exprs):
            return None
        import sympy as sp

        syms = [sp.Symbol(v, real=True) for v in self.variables[: self.dim]]
        J = sp.Matrix([[sp.diff(f, s) for s in syms] for f in exprs])
        div = sp.simplify(sum(J[i, i] for i in range(self.dim)))
        if self.dim == 2:
            curl = sp.simplify(J[1, 0] - J[0, 1])
        else:
            curl = sp.Matrix([sp.simplify(J[2, 1] - J[1, 2]), sp.simplify(J[0, 2] - J[2, 0]),
                              sp.simplify(J[1, 0] - J[0, 1])])
        return {"F": sp.Matrix(exprs), "J": J.applyfunc(sp.simplify), "div": div, "curl": curl}


@dataclass
class ScalarField:
    """f(x, y[, z], t)."""

    text: str
    dim: int

    def __post_init__(self) -> None:
        self.expr = compile_expr(self.text, VARS_2D if self.dim == 2 else VARS_3D)

    def __call__(self, *coords, t: float = 0.0) -> np.ndarray:
        coords = [np.asarray(c, dtype=float) for c in coords]
        shape = np.broadcast_shapes(*[c.shape for c in coords])
        return self.expr(*coords, np.full(shape, float(t)))

    def gradient_grid(self, grids, t: float = 0.0, h: float = 1e-4) -> np.ndarray:
        out = []
        for i in range(self.dim):
            plus = [g + (h if k == i else 0) for k, g in enumerate(grids)]
            minus = [g - (h if k == i else 0) for k, g in enumerate(grids)]
            out.append((self(*plus, t=t) - self(*minus, t=t)) / (2 * h))
        return np.stack(out)

    def laplacian_grid(self, grids, t: float = 0.0, h: float = 1e-3) -> np.ndarray:
        centre = self(*grids, t=t)
        total = np.zeros_like(centre)
        for i in range(self.dim):
            plus = [g + (h if k == i else 0) for k, g in enumerate(grids)]
            minus = [g - (h if k == i else 0) for k, g in enumerate(grids)]
            total += (self(*plus, t=t) - 2 * centre + self(*minus, t=t)) / (h * h)
        return total

    def symbolic(self):
        f = self.expr.to_sympy()
        if f is None:
            return None
        import sympy as sp

        syms = [sp.Symbol(v, real=True) for v in self.expr.variables[: self.dim]]
        grad = sp.Matrix([sp.simplify(sp.diff(f, s)) for s in syms])
        lap = sp.simplify(sum(sp.diff(f, s, 2) for s in syms))
        hess = sp.hessian(f, syms).applyfunc(sp.simplify)
        return {"f": f, "grad": grad, "lap": lap, "hess": hess}


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def rk4_paths(rhs: Callable[[np.ndarray, float], np.ndarray], start: np.ndarray,
              t0: float, t1: float, steps: int, bound: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Integrate many points at once with classical Runge–Kutta.

    ``start`` has shape (n_points, dim); ``rhs(P, t)`` returns the same shape.
    Returns times (steps+1,) and paths (steps+1, n_points, dim).  Points that
    leave ``|coord| > bound`` (or blow up) are frozen as NaN from then on.
    """
    dt = (t1 - t0) / steps
    P = np.array(start, dtype=float)
    out = np.empty((steps + 1,) + P.shape)
    out[0] = P
    times = t0 + dt * np.arange(steps + 1)
    with np.errstate(all="ignore"):
        for k in range(steps):
            t = times[k]
            k1 = rhs(P, t)
            k2 = rhs(P + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = rhs(P + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = rhs(P + dt * k3, t + dt)
            P = P + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bad = ~np.all(np.isfinite(P), axis=1)
            if bound is not None:
                bad |= np.any(np.abs(P) > bound, axis=1)
            P[bad] = np.nan
            out[k + 1] = P
    return times, out


def field_rhs(F: VectorField, frozen_time: float | None = None):
    """Turn a VectorField into an ODE right-hand side for :func:`rk4_paths`."""

    def rhs(P: np.ndarray, t: float) -> np.ndarray:
        tt = t if frozen_time is None else frozen_time
        return F(*P.T, t=tt).T

    return rhs


def streamlines(F: VectorField, seeds: np.ndarray, t: float, length: float,
                steps: int = 60, bound: float | None = None) -> list[np.ndarray]:
    """Curves tangent to F at the frozen time ``t``, traced both ways from each seed."""

    def unit_rhs(P, _t):
        V = F(*P.T, t=t).T
        n = np.linalg.norm(V, axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            return np.where(n > 1e-12, V / n, 0.0)

    _, fwd = rk4_paths(unit_rhs, seeds, 0, length / 2, steps // 2, bound)
    _, bwd = rk4_paths(lambda P, s: -unit_rhs(P, s), seeds, 0, length / 2, steps // 2, bound)
    lines = []
    for i in range(seeds.shape[0]):
        pts = np.vstack([bwd[::-1, i], fwd[1:, i]])
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) > 1:
            lines.append(pts)
    return lines


# ---------------------------------------------------------------------------
# Fixed points of autonomous systems
# ---------------------------------------------------------------------------

@dataclass
class FixedPoint:
    point: np.ndarray
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    kind: str


def classify_fixed_point(J: np.ndarray) -> str:
    if J.shape == (2, 2):
        return classify_linear_system_2d(J)
    ev = np.linalg.eigvals(J)
    re = ev.real
    rot = " with rotation" if np.any(np.abs(ev.imag) > 1e-9) else ""
    if np.all(re < -1e-9):
        return "Stable (sink)" + rot
    if np.all(re > 1e-9):
        return "Unstable (source)" + rot
    if np.any(re > 1e-9) and np.any(re < -1e-9):
        return "Saddle" + rot
    return "Non-hyperbolic (linearisation inconclusive)"


def find_fixed_points(F: VectorField, bounds: Sequence[float] | float, t: float = 0.0, seeds_per_axis: int = 7,
                      tol: float = 1e-8) -> list[FixedPoint]:
    """Newton's method from a grid of seeds inside ``bounds`` (xmin, xmax, ymin, ...), de-duplicated."""
    if np.isscalar(bounds):
        bounds = [-bounds, bounds] * F.dim
    lo = np.array(bounds[0::2], float)
    hi = np.array(bounds[1::2], float)
    pad = 0.05 * (hi - lo)
    axes = [np.linspace(lo[k], hi[k], seeds_per_axis) for k in range(F.dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    seeds = np.stack([m.ravel() for m in mesh], axis=1)
    found: list[np.ndarray] = []
    for p in seeds:
        x = p.copy()
        for _ in range(40):
            fx = F.at(x, t)
            if not np.all(np.isfinite(fx)):
                break
            J = F.jacobian(x, t)
            try:
                dx = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -fx, rcond=None)[0]
            x = x + dx
            if np.linalg.norm(dx) < tol:
                break
        if (np.all(np.isfinite(x)) and np.linalg.norm(F.at(x, t)) < 1e-6
                and np.all(x >= lo - pad) and np.all(x <= hi + pad)
                and not any(np.linalg.norm(x - q) < 1e-4 for q in found)):
            found.append(x)
    result = []
    for x in found:
        x = np.where(np.abs(x) < 1e-9, 0.0, x)
        J = F.jacobian(x, t)
        result.append(FixedPoint(x, J, np.linalg.eigvals(J), classify_fixed_point(J)))
    return result


# ---------------------------------------------------------------------------
# Parametric curves r(t): motion of a point
# ---------------------------------------------------------------------------

@dataclass
class ParametricCurve:
    components: Sequence[str]

    def __post_init__(self) -> None:
        self.exprs = [compile_expr(c, ("t",)) for c in self.components]

    @property
    def dim(self) -> int:
        return len(self.components)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.stack([e(t) for e in self.exprs], axis=-1)

    def kinematics(self, t: np.ndarray, h: float = 1e-4) -> dict[str, np.ndarray]:
        """Position, velocity, acceleration, Frenet frame, curvature and torsion."""
        t = np.asarray(t, dtype=float)
        r = self(t)
        v = (self(t + h) - self(t - h)) / (2 * h)
        a = (self(t + h) - 2 * r + self(t - h)) / (h * h)
        h3 = 1e-3
        jerk = (self(t + 2 * h3) - 2 * self(t + h3) + 2 * self(t - h3) - self(t - 2 * h3)) / (2 * h3 ** 3)
        pad = lambda M: M if M.shape[-1] == 3 else np.concatenate([M, np.zeros(M.shape[:-1] + (1,))], -1)  # noqa: E731
        v3, a3, j3 = pad(v), pad(a), pad(jerk)
        speed = np.linalg.norm(v3, axis=-1)
        cross = np.cross(v3, a3)
        cross_n = np.linalg.norm(cross, axis=-1)
        with np.errstate(all="ignore"):
            T = v3 / speed[..., None]
            B = cross / cross_n[..., None]
            N = np.cross(B, T)
            curvature = cross_n / speed ** 3
            torsion = np.einsum("...i,...i", cross, j3) / cross_n ** 2
        return {"r": r, "v": v, "a": a, "speed": speed, "T": T, "N": N, "B": B,
                "curvature": curvature, "torsion": torsion}
