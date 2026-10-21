"""Matrix decompositions as a sequence of simple moves: rotate, stretch, shear, swap."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..core import linalg as la
from ..presets import MATRICES
from ..ui import dimension_picker, fmt, header, latex_matrix, matrix_editor, show, tip
from ..viz import plot as P
from ..viz.scenes import SceneOptions, factor_path, transform_figure

METHODS = {
    "SVD  A = UΣVᵀ": "Every matrix = rotate (Vᵀ), stretch along the axes (Σ), rotate again (U).",
    "Eigendecomposition  A = PDP⁻¹": "Switch to eigen-coordinates (P⁻¹), stretch each axis by λ (D), switch back (P).",
    "Polar  A = QS": "Stretch along perpendicular directions (symmetric S), then rotate (Q).",
    "QR  A = QR": "Triangular shear + scale (R), then a rotation/reflection (Q).",
    "LU  A = PᵀLU": "Upper-triangular (U), then lower-triangular (L), then a row swap (Pᵀ) — Gaussian elimination.",
}


def _factors(A: np.ndarray, method: str):
    """Return (factors in the order they are applied, their names, extra vectors to draw) or an error."""
    n = A.shape[0]
    if method.startswith("SVD"):
        U, s, Vt = la.svd(A)
        S = np.diag(s)
        vecs = [(Vt[k], P.BASIS_COLORS[k], f"v{k + 1} → σ{k + 1}u{k + 1} (σ{k + 1} = {fmt(s[k])})") for k in range(n)]
        return [Vt, S, U], ["Vᵀ", "Σ", "U"], vecs, {"U": U, "Σ": S, "Vᵀ": Vt}
    if method.startswith("Eigen"):
        res = la.diagonalize(A)
        if res is None:
            return "This matrix is not diagonalizable (not enough independent eigenvectors)."
        Pm, D = res
        if np.iscomplexobj(Pm):
            return "The eigenvalues are complex, so this decomposition has no real picture. Try SVD or polar."
        Pm = Pm / np.linalg.norm(Pm, axis=0)
        vecs = [(Pm[:, k], P.BASIS_COLORS[k], f"eigenvector {k + 1} (λ = {fmt(D[k, k])})") for k in range(n)]
        return [np.linalg.inv(Pm), D, Pm], ["P⁻¹", "D", "P"], vecs, {"P": Pm, "D": D, "P⁻¹": np.linalg.inv(Pm)}
    if method.startswith("Polar"):
        U, s, Vt = la.svd(A)
        Q = U @ Vt
        S = Vt.T @ np.diag(s) @ Vt
        return [S, Q], ["S", "Q"], [], {"Q": Q, "S": S}
    if method.startswith("QR"):
        Q, R = la.qr_decomposition(A)
        return [R, Q], ["R", "Q"], [], {"Q": Q, "R": R}
    Pm, L, U = la.lu_decomposition(A)
    return [U, L, Pm.T], ["U", "L", "Pᵀ"], [], {"Pᵀ": Pm.T, "L": L, "U": U}


def render() -> None:
    header("Decompositions", "Break any matrix into simple moves and watch them happen one after another.")
    dim = dimension_picker("dec_dim")
    left, right = st.columns([1, 2.3])
    with left:
        method = st.radio("Decomposition", list(METHODS), key="dec_method")
        st.caption(METHODS[method])
        A = matrix_editor("dec_A", dim, dim, MATRICES[dim]["Stretch along eigenvectors" if dim == 2 else "Upper triangular"],
                          MATRICES[dim], label="A")
        span = st.slider("View size", 1.5, 10.0, 4.0 if dim == 2 else 2.5, 0.5, key=f"dec_span{dim}")
    if A is None:
        return
    out = _factors(A, method)
    if isinstance(out, str):
        with right:
            st.warning(out)
        return
    factors, names, vecs, show_mats = out
    o = SceneOptions(span=span, circle=True, shape=dim == 2 or not method.startswith("SVD"), vectors=vecs)

    def label(t: float) -> str:
        k = min(int(t), len(names) - 1)
        return f"{t:.2f}  (applying {names[k]})"

    ts = np.linspace(0, len(factors), 20 * len(factors) + 1)
    fig = transform_figure(dim, factor_path(factors), ts, o, label=label)
    with right:
        show(fig, key="dec_fig")
    st.markdown("**Order of moves:** " + "  →  ".join(f"`{n}`" for n in names)
                + f"   (so A = {' '.join(reversed(names))})")
    cols = st.columns(len(show_mats) + 1)
    for c, (name, M) in zip(cols, show_mats.items()):
        with c:
            latex_matrix(M, f"{name.replace('ᵀ', '^T').replace('⁻¹', '^{-1}')} = ")
    with cols[-1]:
        product = np.eye(dim)
        for F in factors:
            product = F @ product
        st.markdown(f"Check: product = A? {'✅' if np.allclose(product, A) else '❌'}")
    if method.startswith("SVD"):
        s = np.diag(show_mats["Σ"])
        st.markdown(f"Singular values: {', '.join(fmt(x) for x in s)}. The unit circle/sphere becomes an ellipse/"
                    f"ellipsoid whose half-axes are exactly these. Condition number σ_max/σ_min = "
                    f"{fmt(s[0] / s[-1]) if s[-1] > 1e-12 else '∞'}.")
        tip("The coloured arrows vᵢ are the input directions that end up as the ellipse's axes. Dropping the "
            "smallest σ gives the best low-rank approximation — the idea behind image compression and PCA.")
    else:
        tip("Rotations (orthogonal factors) are animated as true turns; other factors slide straight from the "
            "identity. Drag the slider to stop between moves.")
