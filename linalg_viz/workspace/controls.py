"""The control panel (sidebar): every layer of the single workspace graph is switched on here."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from ..core import linalg as la
from ..core.fields import ParametricCurve, ScalarField, VectorField
from ..core.parsing import ExpressionError, eval_number
from ..presets import CURVES, FIELDS, MATRICES, SCALARS
from ..ui import matrix_editor, set_matrix
from ..viz import plot as P

FORMULA_HELP = ("Use x, y, z and time t. Functions: sin cos tan exp log sqrt abs atan2 sinh cosh tanh "
                "min max where(cond, a, b). Powers: x^2. 2x means 2*x.")

SPLITS = {
    "one smooth move": "Rotations turn, shears slide, stretches grow — straight from I to A.",
    "SVD: rotate → stretch → rotate": "A = UΣVᵀ: turn (Vᵀ), stretch along the axes (Σ), turn again (U).",
    "Eigen: P⁻¹ → D → P": "Switch to eigen-coordinates, stretch each axis by λ, switch back.",
    "Polar: stretch → rotate": "A = QS: stretch along perpendicular directions (S), then turn (Q).",
    "QR: R → Q": "Triangular shear + scale (R), then a rotation/reflection (Q).",
    "LU: U → L → Pᵀ": "Gaussian elimination: upper-triangular, lower-triangular, then a row swap.",
}

VECTOR_EXTRAS = ["sum", "linear combination", "span", "projection v₁→v₂", "cross product", "Gram–Schmidt",
                 "power method", "coordinates in basis B"]
EIGEN_EXTRAS = ["eigenvector lines", "column space", "null space", "row space", "left null space"]
SEEDINGS = ["blob (disc / ball)", "grid", "random", "line"]


@dataclass
class Step:
    name: str
    matrix: np.ndarray


@dataclass
class NamedVector:
    name: str
    v: np.ndarray
    color: str


@dataclass
class Settings:
    dim: int = 2
    span: float = 4.0
    mode: str = "transform"  # what ▶ Play animates: "transform" or "time"
    labels: bool = True
    # transformation
    tf_on: bool = True
    A: np.ndarray | None = None
    B: np.ndarray | None = None
    steps: list[Step] = field(default_factory=list)
    split: str = "one smooth move"
    grid: bool = True
    shape: bool = True
    circle: bool = False
    basis_arrows: bool = True
    fan: bool = False
    # vectors
    vectors: list[NamedVector] = field(default_factory=list)
    ride: bool = True
    vec_extras: list[str] = field(default_factory=list)
    coeffs: list[float] = field(default_factory=list)
    # eigen & subspaces
    eig_extras: list[str] = field(default_factory=list)
    # Ax = b
    sys_on: bool = False
    b: np.ndarray | None = None
    sys_extras: list[str] = field(default_factory=list)
    # basis
    basis_on: bool = False
    basis: np.ndarray | None = None
    # vector field
    field_on: bool = False
    F: VectorField | None = None
    field_source: str = "formula"
    field_name: str = ""
    arrows: bool = True
    normalise: bool = True
    stream: bool = False
    density: int = 20
    background: str = "none"
    # scalar
    scalar_on: bool = False
    f: ScalarField | None = None
    contours: bool = True
    grad_arrows: bool = False
    # particles
    particles_on: bool = False
    seeding: str = SEEDINGS[0]
    count: int = 120
    centre: np.ndarray | None = None
    radius: float = 0.6
    trail: int = 12
    fixed_points: bool = False
    # probe
    probe_on: bool = False
    p: np.ndarray | None = None
    probe_r: float = 0.4
    probe_extras: list[str] = field(default_factory=list)
    # curve
    curve_on: bool = False
    curve: ParametricCurve | None = None
    c0: float = 0.0
    c1: float = 1.0
    curve_extras: list[str] = field(default_factory=list)
    # time
    t0: float = 0.0
    t1: float = 10.0
    frames: int = 60
    t_now: float = 0.0
    space_flow: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def window(self) -> tuple[float, ...]:
        return tuple([-self.span, self.span] * self.dim)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pills(label: str, options: list[str], default: list[str], key: str) -> list[str]:
    return list(st.pills(label, options, selection_mode="multi", default=default, key=key) or [])


def _matrix_steps(A: np.ndarray, split: str) -> tuple[list[Step], str | None]:
    n = A.shape[0]
    if split.startswith("SVD"):
        U, s, Vt = la.svd(A)
        return [Step("Vᵀ", Vt), Step("Σ", np.diag(s)), Step("U", U)], None
    if split.startswith("Eigen"):
        res = la.diagonalize(A)
        if res is None or np.iscomplexobj(res[0]):
            return [Step("A", A)], "No real eigendecomposition for this A — played as one move."
        Pm, D = res
        Pm = Pm / np.linalg.norm(Pm, axis=0)
        return [Step("P⁻¹", np.linalg.inv(Pm)), Step("D", D), Step("P", Pm)], None
    if split.startswith("Polar"):
        U, s, Vt = la.svd(A)
        return [Step("S", Vt.T @ np.diag(s) @ Vt), Step("Q", U @ Vt)], None
    if split.startswith("QR"):
        Q, R = la.qr_decomposition(A)
        return [Step("R", R), Step("Q", Q)], None
    if split.startswith("LU"):
        Pm, L, U = la.lu_decomposition(A)
        steps = [Step("U", U), Step("L", L)]
        if not np.allclose(Pm, np.eye(n)):
            steps.append(Step("Pᵀ", Pm.T))
        return steps, None
    return [Step("A", A)], None


def _vector_table(dim: int) -> list[NamedVector]:
    key = f"ws_vec_table{dim}"
    base = pd.DataFrame({"name": ["v", "w"], "x": ["1", "2"], "y": ["2", "-1"], "z": ["1", "0.5"]})
    cols = ["name", "x", "y"] + (["z"] if dim == 3 else [])
    config = {c: st.column_config.TextColumn(c, width="small") for c in cols}
    table = st.data_editor(base[cols], key=key, num_rows="dynamic", hide_index=True, column_config=config,
                           width="stretch")
    out, problems = [], []
    for i, row in enumerate(table.to_dict("records")):
        name = str(row.get("name") or f"v{i + 1}").strip() or f"v{i + 1}"
        entries = [row.get(c) for c in "xyz"[:dim]]
        if all(e is None or str(e).strip() == "" or str(e) == "nan" for e in entries):
            continue
        try:
            v = np.array([eval_number(str(e)) if e is not None and str(e).strip() not in ("", "nan") else 0.0
                          for e in entries])
        except (ExpressionError, ValueError, TypeError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        out.append(NamedVector(name, v, P.color(len(out))))
    for msg in problems:
        st.error(msg)
    return out


def _preset_text_inputs(prefix: str, presets: dict, names: list[str], labels: list[str], default: str,
                        values_of) -> tuple[str, list[str]]:
    """Preset selectbox that fills a group of text inputs (field components, f, r(t)…)."""
    pkey = f"{prefix}_preset"
    keys = [f"{prefix}_{k}" for k in range(len(labels))]

    def apply() -> None:
        name = st.session_state[pkey]
        if name in presets:
            for k, value in zip(keys, values_of(presets[name])):
                st.session_state[k] = value

    if pkey not in st.session_state or any(k not in st.session_state for k in keys):
        st.session_state[pkey] = default
        apply()
    st.selectbox("Preset", names + ["(custom)"], key=pkey, on_change=apply)
    texts = [st.text_input(lab, key=k, help=FORMULA_HELP) for lab, k in zip(labels, keys)]
    return st.session_state[pkey], texts


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _transformation(s: Settings) -> None:
    with st.expander("🔀 Matrix / transformation", expanded=True):
        s.tf_on = st.toggle("Apply the matrix to space", True, key="ws_tf_on")
        default = MATRICES[s.dim]["Rotation 45°" if s.dim == 2 else "Rotate 45° about z"]
        s.A = matrix_editor("ws_A", s.dim, s.dim, default, MATRICES[s.dim], label="A")
        s.split = st.selectbox("Play A as", list(SPLITS), key="ws_split")
        st.caption(SPLITS[s.split])
        then_b = st.checkbox("…then apply a second matrix B (composition BA)", key="ws_then_b")
        if then_b:
            s.B = matrix_editor("ws_B", s.dim, s.dim, MATRICES[s.dim]["Shear (horizontal)" if s.dim == 2
                                                                    else "Shear x by z"],
                                MATRICES[s.dim], label="B")
        undo = st.checkbox("…finally undo everything (apply the inverse)", key="ws_undo")
        shows = _pills("Show", ["grid", "unit square/cube", "unit circle/sphere", "basis î ĵ k̂", "fan of unit vectors"],
                       ["grid", "unit square/cube", "basis î ĵ k̂"], f"ws_tf_show{s.dim}")
        s.grid = "grid" in shows
        s.shape = "unit square/cube" in shows
        s.circle = "unit circle/sphere" in shows
        s.basis_arrows = "basis î ĵ k̂" in shows
        s.fan = "fan of unit vectors" in shows
    if s.A is None:
        s.errors.append("Matrix A has an error.")
        return
    if not s.tf_on:
        return
    steps, note = _matrix_steps(s.A, s.split)
    if note:
        s.errors.append(note)
    if s.B is not None:
        steps.append(Step("B", s.B))
    if undo:
        total = np.eye(s.dim)
        for st_ in steps:
            total = st_.matrix @ total
        if abs(np.linalg.det(total)) < 1e-10:
            s.errors.append("Cannot undo: the transformation squashes space (det = 0), so it has no inverse.")
        else:
            steps.append(Step("inverse", np.linalg.inv(total)))
    s.steps = steps


def _vectors(s: Settings) -> None:
    with st.expander("➡️ Vectors", expanded=True):
        st.caption("Add rows to add vectors. Entries may be expressions like sqrt(2).")
        s.vectors = _vector_table(s.dim)
        s.ride = st.checkbox("Vectors move with the transformation", True, key="ws_ride")
        s.vec_extras = _pills("Show", VECTOR_EXTRAS, [], f"ws_vec_extras{s.dim}")
        if "linear combination" in s.vec_extras and s.vectors:
            s.coeffs = [st.slider(f"c · {nv.name}", -3.0, 3.0, 1.0, 0.1, key=f"ws_c{k}")
                        for k, nv in enumerate(s.vectors[:4])]


def _eigen(s: Settings) -> None:
    with st.expander("🌟 Eigenvectors & subspaces of A"):
        s.eig_extras = _pills("Draw", EIGEN_EXTRAS, [], f"ws_eig{s.dim}")


def _system(s: Settings) -> None:
    with st.expander("🎯 Equations  A x = b"):
        s.sys_on = st.toggle("Solve A x = b", key="ws_sys_on")
        cols = st.columns(s.dim)
        b = []
        for k in range(s.dim):
            text = cols[k].text_input(f"b{k + 1}", ["4", "2", "1"][k], key=f"ws_b{k}{s.dim}")
            try:
                b.append(eval_number(text))
            except (ExpressionError, ValueError) as exc:
                st.error(str(exc))
                b.append(0.0)
        s.b = np.array(b)
        s.sys_extras = _pills("Show", ["row picture", "column picture", "solution"],
                              ["row picture", "solution"], f"ws_sys_show{s.dim}")


def _basis(s: Settings) -> None:
    with st.expander("🧭 Another basis B"):
        s.basis_on = st.toggle("Show basis B and its grid", key="ws_basis_on")
        default = [[2, -1], [1, 1]] if s.dim == 2 else [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
        s.basis = matrix_editor("ws_basis", s.dim, s.dim, default, label="B (basis vectors = columns)",
                                show_latex=False)
        if s.A is not None:
            vals, vecs = np.linalg.eig(s.A)
            ok = np.allclose(vals.imag, 0) and abs(np.linalg.det(vecs)) > 1e-8
            st.button("Use the eigenvectors of A", disabled=not ok, key="ws_eigbasis",
                      on_click=lambda: set_matrix("ws_basis", np.real(vecs) / np.max(np.abs(np.real(vecs)), axis=0)))
        if s.basis is not None and abs(np.linalg.det(s.basis)) < 1e-10:
            st.error("These columns are dependent — not a basis.")
            s.basis = None


def _field(s: Settings) -> None:
    with st.expander("🌬️ Vector field  F(x, y, z, t)"):
        s.field_on = st.toggle("Show a vector field", key="ws_field_on")
        s.field_source = st.radio("Field", ["my formula", "the matrix: F = A·x"], horizontal=True,
                                  key="ws_field_src")
        presets = FIELDS[s.dim]
        names = list(presets)
        labels = [f"F{c}(x, y{', z' if s.dim == 3 else ''}, t) =" for c in "xyz"[: s.dim]]
        name, comps = _preset_text_inputs(f"ws_field{s.dim}", presets, names, labels,
                                          "Rotation (vortex)" if s.dim == 2 else "Helical flow", lambda v: v[0])
        if name in presets:
            st.caption(presets[name][2])
        s.field_name = name
        if s.field_source.startswith("the matrix") and s.A is not None:
            names_ = "xyz"[: s.dim]
            comps = [" + ".join(f"({float(s.A[i, j])!r})*{names_[j]}" for j in range(s.dim)) for i in range(s.dim)]
        try:
            F = VectorField(comps)
            F(*[np.zeros(1)] * s.dim, t=0.0)
            s.F = F
        except (ExpressionError, ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(f"Field: {exc}")
        shows = _pills("Show", ["arrows", "equal-length arrows", "streamlines"], ["arrows", "equal-length arrows"],
                       f"ws_field_show{s.dim}")
        s.arrows = "arrows" in shows
        s.normalise = "equal-length arrows" in shows
        s.stream = "streamlines" in shows
        s.density = st.slider("Arrow density", 5, 40, 20 if s.dim == 2 else 7, key=f"ws_density{s.dim}")
        if s.dim == 2:
            s.background = st.selectbox("Colour the background by", ["none", "strength |F|", "divergence ∇·F",
                                                                     "curl ∇×F"], key="ws_bg")


def _scalar(s: Settings) -> None:
    with st.expander("⛰️ Scalar function  f(x, y, z, t)"):
        s.scalar_on = st.toggle("Show a function f", key="ws_scalar_on")
        presets = SCALARS[s.dim]
        _, (text,) = _preset_text_inputs(f"ws_scalar{s.dim}", presets, list(presets),
                                         [f"f(x, y{', z' if s.dim == 3 else ''}, t) ="], list(presets)[0],
                                         lambda v: [v])
        try:
            f = ScalarField(text, s.dim)
            f(*[np.zeros(1)] * s.dim, t=0.0)
            s.f = f
        except (ExpressionError, ValueError, TypeError) as exc:
            st.error(f"f: {exc}")
        shows = _pills("Show", ["level sets", "gradient ∇f arrows"], ["level sets"], f"ws_scalar_show{s.dim}")
        s.contours = "level sets" in shows
        s.grad_arrows = "gradient ∇f arrows" in shows


def _particles(s: Settings) -> None:
    with st.expander("🌊 Particles (follow the field over time)"):
        s.particles_on = st.toggle("Release particles", key="ws_part_on")
        st.caption("Particles obey dx/dt = F(x, t) using the field from 🌬️ (it does not need to be visible).")
        s.seeding = st.selectbox("Start as", SEEDINGS, key="ws_seed")
        s.count = st.slider("How many", 5, 400, 120 if s.dim == 2 else 60, key=f"ws_count{s.dim}")
        cols = st.columns(s.dim)
        s.centre = np.array([cols[k].number_input(f"centre {'xyz'[k]}", value=[1.5, 0.0, 0.0][k], step=0.25,
                                                  key=f"ws_cen{k}{s.dim}") for k in range(s.dim)])
        s.radius = st.slider("Blob radius / line half-length", 0.05, 5.0, 0.6, 0.05, key="ws_rad")
        s.trail = st.slider("Trail length (frames)", 0, 60, 12, key="ws_trail")
        s.fixed_points = st.checkbox("Mark fixed points (F = 0)", key="ws_fp")


def _probe(s: Settings) -> None:
    with st.expander("🔍 Probe a point (div, curl, gradient, Jacobian)"):
        s.probe_on = st.toggle("Place a probe", key="ws_probe_on")
        s.p = np.array([st.slider(f"probe {a}", -s.span, s.span, min(1.0, s.span) * [1, 0.5, 0.3][k],
                                  s.span / 100, key=f"ws_p{a}{s.dim}") for k, a in enumerate("xyz"[: s.dim])])
        s.probe_r = st.slider("probe radius", 0.05, s.span / 2, min(0.5, s.span / 4), s.span / 200, key="ws_pr")
        s.probe_extras = _pills("Show", ["flux (divergence)", "paddle wheel (curl)", "local linear map (Jacobian)",
                                         "gradient of f"], ["flux (divergence)", "paddle wheel (curl)"],
                                f"ws_probe_show{s.dim}")


def _curve(s: Settings) -> None:
    with st.expander("🛰️ Moving point  r(t)"):
        s.curve_on = st.toggle("Show a curve r(t)", key="ws_curve_on")
        presets = CURVES[s.dim]
        labels = [f"{c}(t) =" for c in "xyz"[: s.dim]] + ["t from", "t to"]
        _, texts = _preset_text_inputs(f"ws_curve{s.dim}", presets, list(presets), labels, list(presets)[0],
                                       lambda v: list(v[0]) + [str(v[1][0]), str(v[1][1])])
        try:
            s.curve = ParametricCurve(texts[: s.dim])
            s.c0, s.c1 = eval_number(texts[-2]), eval_number(texts[-1])
            if s.c1 <= s.c0:
                raise ExpressionError("'t to' must be larger than 't from'")
            s.curve(np.array([s.c0]))
        except (ExpressionError, ValueError, TypeError) as exc:
            st.error(f"r(t): {exc}")
            s.curve = None
        s.curve_extras = _pills("Show", ["velocity", "acceleration", "T N B frame", "osculating circle"],
                                ["velocity", "acceleration"] + (["T N B frame"] if s.dim == 3 else []),
                                f"ws_curve_show{s.dim}")
        st.caption("The point runs from 't from' to 't to' while ▶ Play animates time.")


def _time(s: Settings) -> None:
    with st.expander("⏱️ Time"):
        s.t0, s.t1 = st.slider("Time range", 0.0, 60.0, (0.0, 10.0), 0.5, key="ws_trange")
        if s.t1 <= s.t0:
            s.t1 = s.t0 + 0.5
        s.frames = st.slider("Animation frames", 10, 150, 60, key="ws_frames")
        s.t_now = st.slider("Time shown when not playing", s.t0, s.t1, s.t0, (s.t1 - s.t0) / 200, key="ws_tnow")
        s.space_flow = st.checkbox("Let all of space flow with x′ = A x  (grid moves by e^{tA})", key="ws_flow")


# ---------------------------------------------------------------------------

def read_controls() -> Settings:
    s = Settings()
    sb = st.sidebar
    with sb:
        st.markdown("## 🧮 Linear Algebra Visualizer")
        c1, c2 = st.columns(2)
        s.dim = 2 if c1.radio("Space", ["2D", "3D"], horizontal=True, key="ws_dim") == "2D" else 3
        s.labels = c2.checkbox("Names on arrows", True, key="ws_labels")
        s.span = st.slider("Zoom (half-width of the view)", 1.0, 30.0, 4.0 if s.dim == 2 else 3.0, 0.5,
                           key=f"ws_span{s.dim}")
        _transformation(s)
        _vectors(s)
        _eigen(s)
        _system(s)
        _basis(s)
        _field(s)
        _scalar(s)
        _particles(s)
        _probe(s)
        _curve(s)
        _time(s)
    return s
