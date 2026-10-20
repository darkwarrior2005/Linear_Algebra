"""Widgets shared by the vector-field pages: define a field, pick a window, build grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from .core.fields import ScalarField, VectorField
from .core.parsing import ExpressionError
from .presets import FIELD_DOMAINS, FIELDS, SCALARS

COMPONENT_NAMES = ("Fx", "Fy", "Fz")

HELP = ("Use x, y" + ", z" + " and t (time). Functions: sin cos tan exp log sqrt abs atan2 sinh cosh tanh "
        "min max where(cond, a, b). Powers: x^2 or x**2.")


@dataclass
class Window:
    bounds: tuple[float, ...]  # (xmin, xmax, ymin, ymax[, zmin, zmax])

    @property
    def dim(self) -> int:
        return len(self.bounds) // 2

    def axis(self, k: int, n: int) -> np.ndarray:
        return np.linspace(self.bounds[2 * k], self.bounds[2 * k + 1], n)

    def mesh(self, n: int):
        axes = [self.axis(k, n) for k in range(self.dim)]
        return axes, np.meshgrid(*axes, indexing="xy")

    @property
    def size(self) -> float:
        return max(self.bounds[2 * k + 1] - self.bounds[2 * k] for k in range(self.dim))

    def contains(self, P: np.ndarray, pad: float = 0.0) -> np.ndarray:
        ok = np.ones(P.shape[0], bool)
        for k in range(self.dim):
            lo, hi = self.bounds[2 * k], self.bounds[2 * k + 1]
            ok &= (P[:, k] >= lo - pad) & (P[:, k] <= hi + pad)
        return ok

    def layout_range(self):
        return list(self.bounds[:4])


def _apply_field_preset(prefix: str, dim: int) -> None:
    name = st.session_state[f"{prefix}_preset{dim}"]
    if name not in FIELDS[dim]:
        return
    comps, span, _ = FIELDS[dim][name]
    for k, c in enumerate(comps):
        st.session_state[f"{prefix}_c{dim}_{k}"] = c
    dom = FIELD_DOMAINS.get(name)
    for k in range(dim):
        lo, hi = (dom[2 * k], dom[2 * k + 1]) if dom and 2 * k + 1 < len(dom) else (-span, span)
        st.session_state[f"{prefix}_lo{dim}_{k}"] = float(lo)
        st.session_state[f"{prefix}_hi{dim}_{k}"] = float(hi)


def field_input(prefix: str, dim: int, default: str | None = None) -> tuple[VectorField | None, Window, str]:
    """Preset picker + one text box per component + plotting window."""
    presets = FIELDS[dim]
    names = list(presets) + ["(custom)"]
    default = default if default in presets else names[0]
    key = f"{prefix}_preset{dim}"
    if key not in st.session_state or f"{prefix}_c{dim}_0" not in st.session_state:
        st.session_state.setdefault(key, default)
        if st.session_state[key] not in presets:
            st.session_state[key] = default
        _apply_field_preset(prefix, dim)
    st.selectbox("Field preset", names, key=key, on_change=_apply_field_preset, args=(prefix, dim),
                 help="Pick an example, then edit the formulas freely.")
    name = st.session_state[key]
    if name in presets:
        st.caption(presets[name][2])
    comps = []
    for k in range(dim):
        comps.append(st.text_input(f"{COMPONENT_NAMES[k]}(x, y{', z' if dim == 3 else ''}, t) =",
                                   key=f"{prefix}_c{dim}_{k}", help=HELP))
    with st.expander("Window", expanded=False):
        bounds = []
        for k, axis in enumerate("xyz"[:dim]):
            c1, c2 = st.columns(2)
            bounds.append(c1.number_input(f"{axis} min", key=f"{prefix}_lo{dim}_{k}", step=0.5, format="%.2f"))
            bounds.append(c2.number_input(f"{axis} max", key=f"{prefix}_hi{dim}_{k}", step=0.5, format="%.2f"))
    for k in range(dim):
        if bounds[2 * k + 1] <= bounds[2 * k]:
            bounds[2 * k + 1] = bounds[2 * k] + 1.0
    window = Window(tuple(float(b) for b in bounds))
    try:
        F = VectorField(comps)
        F(*[np.zeros(1)] * dim, t=0.0)
    except (ExpressionError, ValueError, TypeError, ZeroDivisionError) as exc:
        st.error(f"Could not use this field: {exc}")
        return None, window, name
    return F, window, name


def scalar_input(prefix: str, dim: int) -> ScalarField | None:
    presets = SCALARS[dim]
    names = list(presets) + ["(custom)"]
    pkey, tkey = f"{prefix}_spreset{dim}", f"{prefix}_stext{dim}"

    def apply() -> None:
        if st.session_state[pkey] in presets:
            st.session_state[tkey] = presets[st.session_state[pkey]]

    if pkey not in st.session_state or tkey not in st.session_state:
        st.session_state[pkey] = names[0]
        st.session_state[tkey] = presets[names[0]]
    st.selectbox("Scalar function preset", names, key=pkey, on_change=apply)
    text = st.text_input(f"f(x, y{', z' if dim == 3 else ''}, t) =", key=tkey, help=HELP)
    try:
        f = ScalarField(text, dim)
        f(*[np.zeros(1)] * dim, t=0.0)
    except (ExpressionError, ValueError, TypeError) as exc:
        st.error(f"Could not use this function: {exc}")
        return None
    return f


def time_slider(prefix: str, label: str = "time t", default: float = 0.0, hi: float = 10.0) -> float:
    return st.slider(label, 0.0, hi, default, 0.05, key=f"{prefix}_t")


def latex_or_none(expr) -> str:
    try:
        import sympy as sp

        return sp.latex(expr)
    except Exception:  # noqa: BLE001
        return str(expr)
