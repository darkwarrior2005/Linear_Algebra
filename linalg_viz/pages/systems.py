"""Systems of linear equations: row picture, column picture and row reduction."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import linalg as la
from ..core.parsing import matrix_to_latex
from ..ui import fmt, fmt_vec, header, matrix_editor, show, tip
from ..viz import plot as P

DEFAULTS = {
    (2, 2): ([[1, 2], [3, -1]], [[4], [2]]),
    (3, 2): ([[1, 1], [1, -1], [1, 2]], [[2], [0], [4]]),
    (2, 3): ([[1, 1, 1], [1, -1, 2]], [[3], [1]]),
    (3, 3): ([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [[8], [-11], [-3]]),
}


def _line_2d(a, b, c, span, color, name):
    s = np.linspace(-span * 2, span * 2, 2)
    if abs(b) >= abs(a):
        if abs(b) < 1e-12:
            return go.Scatter(x=[], y=[])
        x, y = s, (c - a * s) / b
    else:
        x, y = (c - b * s) / a, s
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=3), name=name)


def _equation_text(row, rhs, names="xyz"):
    terms = []
    for k, a in enumerate(row):
        if abs(a) < 1e-12:
            continue
        sign = "-" if a < 0 else "+"
        mag = fmt(abs(a))
        terms.append((sign, ("" if mag == "1" else mag) + names[k]))
    if not terms:
        return f"0 = {fmt(rhs)}"
    text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, body in terms[1:]:
        text += f" {sign} {body}"
    return f"{text} = {fmt(rhs)}"


def _row_picture(A, b, sol, span):
    m, n = A.shape
    traces = []
    if n == 2:
        for i in range(m):
            traces.append(_line_2d(A[i, 0], A[i, 1], b[i], span, P.color(i), _equation_text(A[i], b[i])))
        layout = P.layout_2d(span, 560)
    else:
        traces.append(P.axes_3d(span))
        for i in range(m):
            traces.append(P.plane_from_equation(A[i], b[i], span, P.color(i), _equation_text(A[i], b[i]), 0.45))
        layout = P.layout_3d(span, 560)
    if sol.kind == "unique":
        x = sol.particular
        traces.append(P.points_2d(x, "#ffffff", f"solution {fmt_vec(x)}", 13) if n == 2
                      else P.points_3d(x, "#ffffff", f"solution {fmt_vec(x)}", 8))
    elif sol.kind == "infinite":
        x = sol.particular
        if sol.null_basis.shape[1] == 1:
            d = sol.null_basis[:, 0]
            if n == 2:
                e = d * span * 3
                traces.append(go.Scatter(x=[x[0] - e[0], x[0] + e[0]], y=[x[1] - e[1], x[1] + e[1]], mode="lines",
                                         line=dict(color="#ffffff", width=5), name="all solutions (a line)"))
            else:
                traces.append(P.line_3d(d, span, "#ffffff", "all solutions (a line)", 8, None, x))
        elif sol.null_basis.shape[1] == 2 and n == 3:
            N = sol.null_basis
            traces.append(P.plane_through_origin(N[:, 0], N[:, 1], span, "#ffffff", "all solutions (a plane)",
                                                 0.3, offset=x))
    else:
        x = sol.least_squares
        traces.append(P.points_2d(x, "#ffffff", f"best fit (least squares) {fmt_vec(x)}", 12, symbol="x") if n == 2
                      else P.points_3d(x, "#ffffff", f"best fit {fmt_vec(x)}", 7))
    return go.Figure(traces, layout)


def _column_picture(A, b, sol, span):
    m, n = A.shape
    x = sol.particular if sol.particular is not None else sol.least_squares
    arrows = P.arrows_2d if m == 2 else P.arrows_3d
    head = span * (0.045 if m == 2 else 0.07)
    traces = [] if m == 2 else [P.axes_3d(span)]
    origin = np.zeros(m)
    for j in range(n):
        traces += arrows(A[:, j], color=P.color(j), name=f"column {j + 1}", head=head, opacity=0.5)
        traces += arrows(x[j] * A[:, j], origin, color=P.color(j), name=f"{fmt(x[j])} × column {j + 1}", head=head)
        origin = origin + x[j] * A[:, j]
    traces += arrows(b, color="#ffffff", name=f"b = {fmt_vec(b)}", head=head)
    if sol.kind == "none":
        Ax = A @ x
        traces += arrows(Ax, color="#adb5bd", name="closest reachable point Ax̂", head=head)
        if m == 3 and la.rank(A) == 2:
            Q = la.column_space(A)
            traces.append(P.plane_through_origin(Q[:, 0], Q[:, 1], span, "#4dabf7", "column space", 0.2))
        elif m == 2 and la.rank(A) == 1:
            traces.append(P.line_through_origin_2d(la.column_space(A)[:, 0], span, "#4dabf7", "column space"))
    layout = P.layout_2d(span, 560) if m == 2 else P.layout_3d(span, 560)
    return go.Figure(traces, layout)


def render() -> None:
    header("Systems of equations  Ax = b",
           "See every equation as a line/plane (row picture) and as mixing columns (column picture).")
    c = st.columns(3)
    m = c[0].radio("Equations", [2, 3], horizontal=True, key="sys_m")
    n = c[1].radio("Unknowns", [2, 3], horizontal=True, key="sys_n")
    span = c[2].slider("View size", 2.0, 15.0, 6.0, 0.5, key="sys_span")
    A0, b0 = DEFAULTS[(m, n)]
    left, mid = st.columns([2, 1])
    with left:
        A = matrix_editor("sys_A", m, n, A0, label="A (coefficients)", show_latex=False)
    with mid:
        bcol = matrix_editor("sys_b", m, 1, b0, label="b (right side)", show_latex=False)
    if A is None or bcol is None:
        return
    b = bcol[:, 0]
    st.latex(matrix_to_latex(A) + r"\begin{bmatrix}" + r"\\".join("xyz"[:n]) + r"\end{bmatrix} = "
             + matrix_to_latex(bcol))
    sol = la.solve_system(A, b)
    if sol.kind == "unique":
        values = ", ".join(f"{'xyz'[k]} = {fmt(sol.particular[k])}" for k in range(n))
        st.success(f"Exactly one solution: {values}")
    elif sol.kind == "infinite":
        free = sol.null_basis.shape[1]
        st.info(f"Infinitely many solutions ({free} free parameter{'s' if free > 1 else ''}): "
                f"x = {fmt_vec(sol.particular)} + any combination of "
                + ", ".join(fmt_vec(sol.null_basis[:, k]) for k in range(free)))
    else:
        st.warning(f"No exact solution — the equations contradict each other. Best possible (least squares): "
                   f"x̂ = {fmt_vec(sol.least_squares)}, error |Ax̂ − b| = {fmt(sol.residual)}")
    t1, t2, t3 = st.tabs(["📈 Row picture", "➡️ Column picture", "🪜 Row reduction step by step"])
    with t1:
        show(_row_picture(A, b, sol, span), key="sys_row")
        tip("Each equation is a line (2 unknowns) or a plane (3 unknowns). Solutions are where they all "
            "cross. Parallel lines/planes that never meet ⇒ no solution; overlapping ones ⇒ infinitely many.")
    with t2:
        show(_column_picture(A, b, sol, span), key="sys_col")
        tip("Ax is x₁·(column 1) + x₂·(column 2) + …  Solving Ax = b means: how much of each column do I "
            "need to walk to b? If b lies outside the column space there is no way — the best you can do is "
            "the closest point (least squares).")
    with t3:
        aug = np.column_stack([A, b])
        R, steps, pivots = la.rref_steps(aug)
        k = st.slider("Step", 0, len(steps) - 1, len(steps) - 1, key="sys_step") if len(steps) > 1 else 0
        step = steps[k]
        st.markdown(f"**Step {k}: {step.description}**")
        latex = matrix_to_latex(step.matrix)
        # visual separator before the augmented column
        st.latex(latex.replace(r"\begin{bmatrix}", r"\left[\begin{array}{" + "c" * n + "|c}")
                 .replace(r"\end{bmatrix}", r"\end{array}\right]"))
        with st.expander("All steps"):
            for i, s in enumerate(steps):
                st.markdown(f"{i}. {s.description}")
        st.markdown(f"Pivot columns: {', '.join(str(p + 1) for p in pivots if p < n) or 'none'}. "
                    f"rank(A) = {la.rank(A)}, rank([A | b]) = {la.rank(aug)}.")
