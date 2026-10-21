"""Matrix calculator for matrices of any size, with a picture when it is 2×2 or 3×3."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..core import linalg as la
from ..core.parsing import ExpressionError, matrix_to_latex, parse_matrix, vector_to_latex
from ..ui import fmt, header, latex_matrix, show
from ..viz.scenes import SceneOptions, linear_path, transform_figure

ONE = ["determinant", "inverse", "transpose", "rank", "trace", "RREF (row reduction)", "eigenvalues & eigenvectors",
       "SVD", "QR", "LU", "null space", "column space", "power Aⁿ", "matrix exponential e^A", "pseudo-inverse A⁺",
       "norms & condition number", "characteristic polynomial", "k · A", "visualize as a transformation"]
TWO = ["A + B", "A − B", "A · B", "B · A", "solve A x = B (B = right side)", "A ⊙ B (entry-wise)"]


def _read(label: str, key: str, default: str):
    text = st.text_area(label, default, key=key, height=120,
                        help="One row per line, entries separated by spaces or commas. Expressions like sqrt(2) "
                             "or cos(pi/3) are allowed.")
    try:
        return parse_matrix(text)
    except (ExpressionError, ValueError) as exc:
        st.error(str(exc))
        return None


def _one(op: str, A: np.ndarray) -> None:
    m, n = A.shape
    square = m == n
    need_square = {"determinant", "inverse", "trace", "eigenvalues & eigenvectors", "power Aⁿ",
                   "matrix exponential e^A", "characteristic polynomial", "visualize as a transformation"}
    if op in need_square and not square:
        st.error(f"'{op}' needs a square matrix (this one is {m}×{n}).")
        return
    if op == "determinant":
        st.latex(rf"\det A = {fmt(np.linalg.det(A), 6)}")
    elif op == "inverse":
        if abs(np.linalg.det(A)) < 1e-12:
            st.error("det A = 0 — the matrix is singular and has no inverse. Try the pseudo-inverse.")
        else:
            latex_matrix(np.linalg.inv(A), r"A^{-1} = ", 4)
    elif op == "transpose":
        latex_matrix(A.T, "A^T = ")
    elif op == "rank":
        r = la.rank(A)
        st.latex(rf"\operatorname{{rank}} A = {r}, \quad \text{{nullity}} = {n - r}")
    elif op == "trace":
        st.latex(rf"\operatorname{{tr}} A = {fmt(np.trace(A), 6)}")
    elif op.startswith("RREF"):
        R, steps, pivots = la.rref_steps(A)
        latex_matrix(R, r"\operatorname{rref}(A) = ", 4)
        st.caption(f"Pivot columns: {', '.join(str(p + 1) for p in pivots) or 'none'}")
        with st.expander(f"Show all {len(steps) - 1} row operations"):
            for s in steps[1:]:
                st.markdown(f"**{s.description}**")
                st.latex(matrix_to_latex(s.matrix))
    elif op.startswith("eigen"):
        for p in la.eigen_analysis(A):
            vec = np.real(p.vector) if p.is_real else p.vector
            st.latex(rf"\lambda = {fmt(p.value, 4)}\;(\text{{alg}}\ {p.algebraic_multiplicity},\ "
                     rf"\text{{geo}}\ {p.geometric_multiplicity}),\quad v = " + vector_to_latex(vec, 4))
        st.caption("Diagonalizable: " + ("yes" if la.is_diagonalizable(A) else "no"))
    elif op == "SVD":
        U, s, Vt = la.svd(A)
        S = np.zeros((m, n))
        S[: len(s), : len(s)] = np.diag(s)
        st.latex("U = " + matrix_to_latex(U) + r",\ \Sigma = " + matrix_to_latex(S) + r",\ V^T = " + matrix_to_latex(Vt))
    elif op == "QR":
        Q, R = la.qr_decomposition(A)
        st.latex("Q = " + matrix_to_latex(Q) + r",\quad R = " + matrix_to_latex(R))
    elif op == "LU":
        if not square:
            st.error("LU here needs a square matrix.")
            return
        Pm, L, U = la.lu_decomposition(A)
        st.latex(r"PA = LU:\quad P = " + matrix_to_latex(Pm) + r",\ L = " + matrix_to_latex(L) + r",\ U = "
                 + matrix_to_latex(U))
    elif op == "null space":
        N = la.null_space(A)
        if N.shape[1] == 0:
            st.latex(r"N(A) = \{0\}")
        else:
            st.latex(r"N(A) = \operatorname{span}\left\{" + ", ".join(vector_to_latex(N[:, k], 4)
                                                                     for k in range(N.shape[1])) + r"\right\}")
    elif op == "column space":
        C = la.column_space(A)
        st.latex(r"C(A) = \operatorname{span}\left\{" + ", ".join(vector_to_latex(C[:, k], 4)
                                                                 for k in range(C.shape[1])) + r"\right\}")
    elif op == "power Aⁿ":
        k = st.number_input("n", -20, 50, 3, key="calc_pow")
        if k < 0 and abs(np.linalg.det(A)) < 1e-12:
            st.error("Negative powers need an invertible matrix.")
        else:
            latex_matrix(np.linalg.matrix_power(A, int(k)), f"A^{{{int(k)}}} = ", 4)
    elif op.startswith("matrix exponential"):
        latex_matrix(la.matrix_exp(A), "e^{A} = ", 4)
    elif op.startswith("pseudo"):
        latex_matrix(np.linalg.pinv(A), "A^+ = ", 4)
    elif op.startswith("norms"):
        s = np.linalg.svd(A, compute_uv=False)
        cond = s[0] / s[-1] if s[-1] > 1e-14 else np.inf
        st.latex(rf"\|A\|_F = {fmt(np.linalg.norm(A), 4)},\ \|A\|_2 = {fmt(s[0], 4)},\ "
                 rf"\|A\|_1 = {fmt(np.linalg.norm(A, 1), 4)},\ \|A\|_\infty = {fmt(np.linalg.norm(A, np.inf), 4)},\ "
                 rf"\kappa(A) = {fmt(cond, 4) if np.isfinite(cond) else r'\infty'}")
    elif op.startswith("characteristic"):
        st.latex(r"\det(\lambda I - A) = " + la.polynomial_to_text(la.characteristic_polynomial(A)).replace("λ", r"\lambda "))
    elif op == "k · A":
        k = st.number_input("k", value=2.0, key="calc_k")
        latex_matrix(k * A, f"{fmt(k)}A = ")
    elif op.startswith("visualize"):
        if n not in (2, 3):
            st.error("Pictures are available for 2×2 and 3×3 matrices.")
            return
        show(transform_figure(n, linear_path(A), np.linspace(0, 1, 25),
                              SceneOptions(span=4 if n == 2 else 3, circle=n == 2)), key="calc_vis")


def _two(op: str, A: np.ndarray, B: np.ndarray) -> None:
    try:
        if op == "A + B":
            latex_matrix(A + B, "A + B = ")
        elif op == "A − B":
            latex_matrix(A - B, "A - B = ")
        elif op == "A · B":
            latex_matrix(A @ B, "AB = ")
        elif op == "B · A":
            latex_matrix(B @ A, "BA = ")
        elif op.startswith("A ⊙ B"):
            latex_matrix(A * B, r"A \odot B = ")
        else:
            sol = la.solve_system(A, B[:, 0] if B.shape[1] == 1 else B.ravel())
            if sol.kind == "unique":
                st.latex("x = " + vector_to_latex(sol.particular, 4))
            elif sol.kind == "infinite":
                N = sol.null_basis
                st.latex("x = " + vector_to_latex(sol.particular, 4) + " + "
                         + " + ".join(f"t_{k + 1}" + vector_to_latex(N[:, k], 4) for k in range(N.shape[1])))
            else:
                st.warning("No exact solution. Least-squares best fit:")
                st.latex(r"\hat x = " + vector_to_latex(sol.least_squares, 4) + rf",\ \|A\hat x - b\| = {fmt(sol.residual, 4)}")
    except ValueError as exc:
        st.error(f"Sizes do not fit: A is {A.shape[0]}×{A.shape[1]}, B is {B.shape[0]}×{B.shape[1]} ({exc}).")


def render() -> None:
    header("Matrix calculator", "Any size. Type rows on separate lines.")
    a, b = st.columns(2)
    with a:
        A = _read("Matrix A", "calc_A", "2 1 0\n1 3 1\n0 1 4")
    with b:
        B = _read("Matrix B (for two-matrix operations)", "calc_B", "1\n2\n3")
    if A is None:
        return
    st.caption(f"A is {A.shape[0]}×{A.shape[1]}" + (f", B is {B.shape[0]}×{B.shape[1]}" if B is not None else ""))
    t1, t2 = st.tabs(["One matrix (A)", "Two matrices (A, B)"])
    with t1:
        ops = st.multiselect("Operations", ONE, default=["determinant", "rank", "inverse",
                                                         "eigenvalues & eigenvectors"], key="calc_ops")
        for op in ops:
            st.markdown(f"#### {op}")
            _one(op, A)
    with t2:
        if B is None:
            return
        ops2 = st.multiselect("Operations", TWO, default=["A · B", "solve A x = B (B = right side)"], key="calc_ops2")
        for op in ops2:
            st.markdown(f"#### {op}")
            _two(op, A, B)
