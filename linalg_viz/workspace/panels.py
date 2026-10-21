"""Everything that is *not* drawn on the graph: legend chips and information cards."""

from __future__ import annotations

import html

import numpy as np
import streamlit as st

from ..core import linalg as la
from ..core.parsing import format_number as fmt
from ..core.parsing import matrix_to_latex
from ..ui import table
from .controls import Settings
from .scene import Key, Scene, fmt_vec

CHIP_CSS = """
<style>
.lv-legend {display:flex; flex-wrap:wrap; gap:6px 14px; margin:2px 0 10px 0; font-size:0.86rem;}
.lv-chip {display:inline-flex; align-items:center; gap:6px; color:#dee2e6; white-space:nowrap;}
.lv-sw {display:inline-block; width:22px; height:0; border-top:3px solid; border-radius:2px;}
.lv-sw.dash {border-top-style:dashed;}
.lv-sw.area {height:12px; border:2px solid; opacity:0.85;}
.lv-sw.dot {width:10px; height:10px; border:none; border-radius:50%;}
.lv-grad {display:inline-block; width:70px; height:10px; border-radius:3px;}
.lv-small {color:#868e96; font-size:0.78rem;}
</style>
"""


def _chip(k: Key) -> str:
    label = html.escape(k.label)
    if k.kind == "gradient":
        css, lo, hi = (k.detail.split("|") + ["", "", ""])[:3]
        rng = f"<span class='lv-small'>{html.escape(lo)}</span>" if lo else ""
        rng2 = f"<span class='lv-small'>{html.escape(hi)}</span>" if hi else ""
        return f"<span class='lv-chip'>{rng}<span class='lv-grad' style='background:{css}'></span>{rng2} {label}</span>"
    if k.kind == "area":
        sw = f"<span class='lv-sw area' style='border-color:{k.color}; background:{k.color}55'></span>"
    elif k.kind == "dot":
        sw = f"<span class='lv-sw dot' style='background:{k.color}'></span>"
    else:
        sw = f"<span class='lv-sw {'dash' if k.kind == 'dash' else ''}' style='border-color:{k.color}'></span>"
    return f"<span class='lv-chip'>{sw}{label}</span>"


def legend(keys: list[Key]) -> None:
    if not keys:
        return
    seen, chips = set(), []
    for k in keys:
        if (k.color, k.label) not in seen:
            seen.add((k.color, k.label))
            chips.append(_chip(k))
    st.markdown(CHIP_CSS + "<div class='lv-legend'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Information cards
# ---------------------------------------------------------------------------

def _card(title: str):
    box = st.container(border=True)
    box.markdown(f"**{title}**")
    return box


def _transformation_card(s: Settings, scene: Scene) -> None:
    A = s.A
    if A is None:
        return
    with _card("🔀 Matrix A"):
        st.latex("A = " + matrix_to_latex(A))
        det = float(np.linalg.det(A))
        c = st.columns(3)
        c[0].metric("det", fmt(det))
        c[1].metric("rank", la.rank(A))
        c[2].metric("trace", fmt(np.trace(A)))
        for fact in la.describe_transform(A):
            st.markdown(f"- {fact}")
        if s.tf_on and s.steps:
            names = " → ".join(st_.name for st_ in s.steps)
            st.caption(f"▶ plays: {names}. Total matrix applied:")
            st.latex(matrix_to_latex(scene.notes.get("M_final", np.eye(s.dim))))
            if len(s.steps) > 1:
                with st.expander("Each step's matrix"):
                    for st_ in s.steps:
                        st.latex(f"\\text{{{st_.name}}} = " + matrix_to_latex(st_.matrix))
            path = "rotates/slides smoothly (e^{tL})" if la.real_log(A) is not None else \
                "turns its rotation part and slides the stretch/flip part"
            st.caption(f"Animation path: {path}.")


def _eigen_card(s: Settings) -> None:
    A = s.A
    if A is None:
        return
    with _card("🌟 Eigenvalues of A"):
        rows = []
        for p in la.eigen_analysis(A):
            if p.is_real:
                lr = p.value.real
                meaning = ("stays put" if abs(lr - 1) < 1e-9 else "crushed to 0" if abs(lr) < 1e-9 else
                           f"stretched ×{fmt(lr)}" if lr > 0 else f"flipped, ×{fmt(abs(lr))}")
                vec = fmt_vec(np.real(p.vector))
            else:
                meaning = f"turn {fmt(np.degrees(abs(np.angle(p.value))), 1)}° and scale ×{fmt(abs(p.value))}"
                vec = "complex"
            rows.append({"λ": fmt(p.value), "eigenvector": vec, "along it": meaning,
                         "alg/geo": f"{p.algebraic_multiplicity}/{p.geometric_multiplicity}"})
        table(rows)
        st.caption("det(λI − A) = " + la.polynomial_to_text(la.characteristic_polynomial(A))
                   + ("  ·  diagonalizable" if la.is_diagonalizable(A) else "  ·  NOT diagonalizable"))
        if s.dim == 2:
            st.caption("As a velocity rule x′ = Ax: " + la.classify_linear_system_2d(A))


def _vector_card(s: Settings, scene: Scene) -> None:
    if not s.vectors:
        return
    M = scene.notes.get("M_final", np.eye(s.dim))
    with _card("➡️ Vectors"):
        rows = []
        for nv in s.vectors:
            row = {"name": nv.name, "vector": fmt_vec(nv.v), "length": fmt(np.linalg.norm(nv.v))}
            if s.tf_on and s.steps:
                row["after the matrix"] = fmt_vec(M @ nv.v)
            if s.basis is not None and "coordinates in basis B" in s.vec_extras:
                row["in basis B"] = fmt_vec(la.change_of_basis(nv.v, s.basis))
            rows.append(row)
        table(rows)
        V = np.column_stack([nv.v for nv in s.vectors])
        r = la.rank(V)
        st.caption(f"rank {r}: the vectors are {'independent' if r == len(s.vectors) else 'dependent'}; "
                   f"they span {['only 0', 'a line', 'a plane', 'all of 3D'][r] if s.dim == 3 else ['only 0', 'a line', 'the whole plane'][r]}.")
        if len(s.vectors) >= 2:
            a, b = s.vectors[0].v, s.vectors[1].v
            c = st.columns(3)
            c[0].metric(f"{s.vectors[0].name}·{s.vectors[1].name}", fmt(float(np.dot(a, b))))
            c[1].metric("angle", f"{fmt(la.angle_between(a, b), 1)}°")
            if s.dim == 3:
                c[2].metric("|cross|", fmt(np.linalg.norm(np.cross(a, b))))
            else:
                c[2].metric("signed area", fmt(a[0] * b[1] - a[1] * b[0]))
        if "linear combination" in s.vec_extras and s.coeffs:
            cs = np.array(s.coeffs[: len(s.vectors)])
            combo = V[:, : len(cs)] @ cs
            st.caption("c₁v₁ + c₂v₂ + … = " + fmt_vec(combo))
        if "power method" in s.vec_extras and "power" in scene.notes:
            seq = scene.notes["power"]
            w = s.A @ seq[-1]
            st.caption(f"Power method → direction {fmt_vec(seq[-1])}, λ ≈ {fmt(float(np.dot(seq[-1], w)))}")


def _system_card(s: Settings, scene: Scene) -> None:
    sol = scene.notes.get("system")
    if sol is None:
        return
    with _card("🎯 A x = b"):
        if sol.kind == "unique":
            st.success(f"One solution: x = {fmt_vec(sol.particular)}")
        elif sol.kind == "infinite":
            st.info(f"Infinitely many: x = {fmt_vec(sol.particular)} + any mix of "
                    + ", ".join(fmt_vec(sol.null_basis[:, k]) for k in range(sol.null_basis.shape[1])))
        else:
            st.warning(f"No solution. Best fit x̂ = {fmt_vec(sol.least_squares)}, error {fmt(sol.residual)}")
        R, steps, _ = la.rref_steps(np.column_stack([s.A, s.b]))
        with st.expander(f"Row reduction ({len(steps) - 1} steps)"):
            for k, stp in enumerate(steps):
                st.markdown(f"{k}. {stp.description}")
                st.latex(matrix_to_latex(stp.matrix))


def _subspace_card(s: Settings) -> None:
    if s.A is None:
        return
    chosen = [x for x in s.eig_extras if "space" in x]
    if not chosen:
        return
    fn = {"column space": la.column_space, "null space": la.null_space, "row space": la.row_space,
          "left null space": la.left_null_space}
    with _card("🧊 Subspaces of A"):
        st.caption(f"rank {la.rank(s.A)} + nullity {s.dim - la.rank(s.A)} = {s.dim}")
        for name in chosen:
            Q = fn[name](s.A)
            st.markdown(f"{name} (dim {Q.shape[1]}): " + (", ".join(fmt_vec(Q[:, k]) for k in range(Q.shape[1]))
                                                         or "{0}"))


def _field_card(s: Settings, scene: Scene) -> None:
    F = s.F
    if F is None or not (s.field_on or s.particles_on or s.probe_on):
        return
    with _card("🌬️ Vector field"):
        st.caption("F = (" + ", ".join(e.text for e in F.exprs) + ")" + ("  — changes with time" if F.time_dependent
                                                                          else ""))
        sym = F.symbolic()
        if sym is not None:
            import sympy as sp

            st.latex(r"\nabla\cdot F = " + sp.latex(sym["div"]))
            st.latex((r"\text{curl} = " + sp.latex(sym["curl"])) if s.dim == 2 else
                     (r"\nabla\times F = " + sp.latex(sym["curl"].T)))
        fps = scene.notes.get("fixed_points")
        if fps:
            table([{"fixed point": fmt_vec(fp.point), "type": fp.kind,
                    "eigenvalues of J": ", ".join(fmt(v) for v in fp.eigenvalues)} for fp in fps])
        part = scene.notes.get("particles")
        if part and part[2]:
            times, paths, n_ring = part

            def area(Pts):
                x, y = Pts[:, 0], Pts[:, 1]
                return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

            a0, a1 = area(paths[0, :n_ring]), area(paths[-1, :n_ring])
            if np.isfinite(a1):
                st.caption(f"Blob area: {fmt(a0)} at t = {fmt(times[0])} → {fmt(a1)} at t = {fmt(times[-1])}")


def _probe_card(s: Settings, scene: Scene) -> None:
    if not s.probe_on or s.p is None:
        return
    p, t = s.p, s.t_now
    with _card(f"🔍 Probe at {fmt_vec(p)}  (t = {fmt(t)})"):
        if s.F is not None:
            c = st.columns(3)
            c[0].metric("F(p)", fmt_vec(s.F.at(p, t)))
            c[1].metric("divergence", fmt(s.F.divergence_at(p, t)))
            curl = s.F.curl_at(p, t)
            c[2].metric("curl", fmt(curl) if s.dim == 2 else fmt_vec(curl))
            J = s.F.jacobian(p, t)
            st.latex("J = " + matrix_to_latex(J) + r"\quad \operatorname{tr}J = \nabla\cdot F")
            st.caption("eigenvalues of J: " + ", ".join(fmt(v) for v in np.linalg.eigvals(J)))
        if s.f is not None:
            g = s.f.gradient_grid([np.array(v) for v in p], t).ravel()
            lap = s.f.laplacian_grid([np.array(v) for v in p], t)
            st.caption(f"f(p) = {fmt(float(s.f(*p, t=t)))}, ∇f = {fmt_vec(g)}, ∇²f = {fmt(float(lap))}")


def _curve_card(s: Settings) -> None:
    if not s.curve_on or s.curve is None:
        return
    u = s.c0 + (s.t_now - s.t0) / (s.t1 - s.t0) * (s.c1 - s.c0)
    K = s.curve.kinematics(np.array([u]))
    with _card(f"🛰️ r(t) at parameter {fmt(u)}"):
        c = st.columns(3)
        c[0].metric("position", fmt_vec(K["r"][0]))
        c[1].metric("speed", fmt(K["speed"][0]))
        c[2].metric("curvature", fmt(K["curvature"][0]))
        st.caption(f"velocity {fmt_vec(K['v'][0])}, acceleration {fmt_vec(K['a'][0])}"
                   + (f", torsion {fmt(K['torsion'][0])}" if s.dim == 3 else ""))


def info(s: Settings, scene: Scene) -> None:
    cards = [lambda: _transformation_card(s, scene), lambda: _eigen_card(s), lambda: _vector_card(s, scene),
             lambda: _system_card(s, scene), lambda: _subspace_card(s), lambda: _field_card(s, scene),
             lambda: _probe_card(s, scene), lambda: _curve_card(s)]
    cols = st.columns(3)
    for k, card in enumerate(cards):
        with cols[k % 3]:
            card()
