"""Linear-algebra computations used by the visual pages.

Everything here is plain NumPy so it can be unit tested without a UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parsing import format_number

TOL = 1e-9


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def interpolate_matrix(A: np.ndarray, t: float) -> np.ndarray:
    """Straight-line path from the identity (t=0) to ``A`` (t=1)."""
    n = A.shape[0]
    return (1 - t) * np.eye(n) + t * A


def _sqrtm(A: np.ndarray, iters: int = 60) -> np.ndarray:
    """Principal square root by the Denman–Beavers iteration."""
    Y, Z = A.astype(float), np.eye(A.shape[0])
    for _ in range(iters):
        Y_next = 0.5 * (Y + np.linalg.inv(Z))
        Z = 0.5 * (Z + np.linalg.inv(Y))
        if np.allclose(Y_next, Y, atol=1e-14, rtol=1e-13):
            return Y_next
        Y = Y_next
    return Y


def real_log(A: np.ndarray) -> np.ndarray | None:
    """Real principal matrix logarithm, or ``None`` when A has none.

    A real logarithm L (with e^L = A) exists when A has no eigenvalue on the closed
    negative real axis.  Then t ↦ e^{tL} is the most natural way to "grow" A out
    of the identity: rotations turn, shears slide, stretches grow exponentially.
    """
    n = A.shape[0]
    ev = np.linalg.eigvals(A)
    if np.any(np.abs(ev) < 1e-9) or np.any((ev.real <= 0) & (np.abs(ev.imag) < 1e-9)):
        return None
    try:
        X, k = A.astype(float), 0
        while np.linalg.norm(X - np.eye(n), ord=2) > 0.2 and k < 40:
            X, k = _sqrtm(X), k + 1
        E = X - np.eye(n)
        L, term = np.zeros_like(E), np.eye(n)
        for j in range(1, 40):
            term = term @ E
            L = L + ((-1) ** (j + 1)) * term / j
        L = L * 2 ** k
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(L)) or not np.allclose(matrix_exp(L), A, atol=1e-7, rtol=1e-6):
        return None
    return L


def rotation_power(R: np.ndarray, t: float) -> np.ndarray:
    """Rᵗ for a proper rotation: the same turn, t times as far (works through 180°)."""
    n = R.shape[0]
    if n == 2:
        th = t * np.arctan2(R[1, 0], R[0, 0])
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if n == 3:
        th = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        if th < 1e-10:
            return np.eye(3)
        if np.sin(th) < 1e-4:  # (almost) a half turn: axis is the fixed direction
            axis = null_space(R - np.eye(3), tol=1e-4)[:, 0]
        else:
            axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        a = t * th
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return interpolate_matrix(R, t)


def transform_path(A: np.ndarray):
    """A smooth path M(t) from I (t=0) to A (t=1) that rotates instead of shrinking.

    * If A has a real logarithm L: M(t) = e^{tL} (rotations turn rigidly, shears slide,
      rotation+scaling spirals).
    * Otherwise (reflections, projections, 180° turns…): polar split A = R·S into a
      rotation R (turned with :func:`rotation_power`) and a stretch S (slid linearly,
      which is where any flip or squash happens).
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    I = np.eye(n)
    L = real_log(A)
    if L is not None:
        return lambda t: matrix_exp(t * L)
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    S = Vt.T @ np.diag(s) @ Vt
    if np.linalg.det(R) < 0:
        # Move the mirror into the stretch: use R's own reflection normal if it has one.
        w, V = np.linalg.eig(R)
        k = int(np.argmin(np.abs(w + 1)))
        v = np.real(V[:, k]) if abs(w[k] + 1) < 1e-6 else Vt[-1]
        v = v / np.linalg.norm(v)
        H = I - 2 * np.outer(v, v)
        R, S = R @ H, H @ S
    return lambda t: rotation_power(R, t) @ interpolate_matrix(S, t)


def rank(A: np.ndarray, tol: float = 1e-8) -> int:
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s.max())))


def null_space(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis of the null space as columns."""
    _, s, vt = np.linalg.svd(A)
    r = int(np.sum(s > tol * max(1.0, s.max() if s.size else 1.0)))
    return vt[r:].T.copy()


def column_space(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis of the column space as columns."""
    u, s, _ = np.linalg.svd(A)
    r = int(np.sum(s > tol * max(1.0, s.max() if s.size else 1.0)))
    return u[:, :r].copy()


def row_space(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return column_space(A.T, tol)


def left_null_space(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return null_space(A.T, tol)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < TOL or nv < TOL:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def project_onto_vector(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    denom = float(np.dot(onto, onto))
    if denom < TOL:
        return np.zeros_like(v)
    return np.dot(v, onto) / denom * onto


def project_onto_subspace(v: np.ndarray, basis_columns: np.ndarray) -> np.ndarray:
    """Orthogonal projection of ``v`` onto the span of the given columns."""
    Q = column_space(basis_columns)
    return Q @ (Q.T @ v)


def projection_matrix(basis_columns: np.ndarray) -> np.ndarray:
    Q = column_space(basis_columns)
    return Q @ Q.T


# ---------------------------------------------------------------------------
# Describing a transformation in words
# ---------------------------------------------------------------------------

def describe_transform(A: np.ndarray) -> list[str]:
    """Plain-language facts about what a square matrix does to space."""
    n = A.shape[0]
    facts: list[str] = []
    det = float(np.linalg.det(A))
    r = rank(A)
    unit = "area" if n == 2 else "volume"
    if abs(det) < 1e-9:
        dim_word = {0: "a single point", 1: "a line", 2: "a plane"}.get(r, f"a {r}-D subspace")
        facts.append(f"Squashes all of space onto {dim_word} (rank {r}) — det = 0, so it cannot be undone.")
    else:
        facts.append(f"Scales every {unit} by |det| = {format_number(abs(det))}.")
        if det < 0:
            facts.append("Flips orientation (det < 0): space is mirrored.")
    if np.allclose(A, np.eye(n)):
        facts.append("This is the identity: nothing moves.")
        return facts
    if np.allclose(A, A[0, 0] * np.eye(n)):
        facts.append(f"Uniform scaling by {format_number(A[0, 0])} in every direction.")
    if np.allclose(A.T @ A, np.eye(n), atol=1e-6):
        facts.append("Orthogonal: lengths and angles are preserved (a pure rotation"
                     + (" + reflection" if det < 0 else "") + ").")
        if n == 2 and det > 0:
            facts.append(f"Rotation by {format_number(np.degrees(np.arctan2(A[1, 0], A[0, 0])), 2)}°.")
    if np.allclose(A, A.T):
        facts.append("Symmetric: it stretches along perpendicular axes (its eigenvectors).")
    if np.allclose(A @ A, A) and r < n:
        facts.append("Projection: applying it twice is the same as once (A² = A).")
    if np.allclose(np.triu(A), A) or np.allclose(np.tril(A), A):
        diag = np.diag(A)
        if np.allclose(diag, 1) and not np.allclose(A, np.eye(n)):
            facts.append("Shear: slides layers of space parallel to an axis; area is unchanged.")
    eig = np.linalg.eigvals(A)
    if np.any(np.abs(eig.imag) > 1e-9):
        facts.append("Has complex eigenvalues: there is a rotation component, so some directions are never kept.")
    return facts


# ---------------------------------------------------------------------------
# Eigen analysis
# ---------------------------------------------------------------------------

@dataclass
class EigenPair:
    value: complex
    vector: np.ndarray  # complex in general
    algebraic_multiplicity: int
    geometric_multiplicity: int

    @property
    def is_real(self) -> bool:
        return abs(self.value.imag) < 1e-9 and np.allclose(self.vector.imag, 0, atol=1e-9)


def _group_eigenvalues(values: np.ndarray, tol: float = 1e-6) -> list[tuple[complex, int]]:
    groups: list[list[complex]] = []
    for v in values:
        for g in groups:
            if abs(g[0] - v) < tol:
                g.append(v)
                break
        else:
            groups.append([v])
    return [(complex(np.mean(g)), len(g)) for g in groups]


def eigen_analysis(A: np.ndarray) -> list[EigenPair]:
    """Eigenvalues with real eigenvectors where possible plus multiplicities."""
    values, vectors = np.linalg.eig(A)
    n = A.shape[0]
    pairs: list[EigenPair] = []
    for lam, alg in _group_eigenvalues(values):
        if abs(lam.imag) < 1e-9:
            lam = complex(lam.real, 0)
            basis = null_space(A - lam.real * np.eye(n), tol=1e-6)
            geo = max(basis.shape[1], 1)
            if basis.shape[1] == 0:
                idx = int(np.argmin(np.abs(values - lam)))
                basis = np.real(vectors[:, [idx]])
            for k in range(basis.shape[1]):
                v = basis[:, k].astype(complex)
                # Make the largest component positive for a stable display.
                big = np.argmax(np.abs(v))
                if v[big].real < 0:
                    v = -v
                pairs.append(EigenPair(lam, v, alg, geo))
        else:
            idx = int(np.argmin(np.abs(values - lam)))
            pairs.append(EigenPair(lam, vectors[:, idx], alg, 1))
    pairs.sort(key=lambda p: (-abs(p.value), -p.value.real, -p.value.imag))
    return pairs


def is_diagonalizable(A: np.ndarray) -> bool:
    pairs = eigen_analysis(A)
    groups = {}
    for p in pairs:
        key = (round(p.value.real, 6), round(p.value.imag, 6))
        groups[key] = (p.algebraic_multiplicity, p.geometric_multiplicity)
    return all(alg == geo for alg, geo in groups.values())


def diagonalize(A: np.ndarray):
    """Return (P, D) with A = P D P⁻¹ (complex if needed) or ``None``."""
    values, vectors = np.linalg.eig(A)
    if abs(np.linalg.det(vectors)) < 1e-8:
        return None
    if np.allclose(values.imag, 0) and np.allclose(vectors.imag, 0):
        return np.real(vectors), np.diag(np.real(values))
    return vectors, np.diag(values)


def characteristic_polynomial(A: np.ndarray) -> np.ndarray:
    """Coefficients of det(λI − A), highest power first."""
    return np.real_if_close(np.poly(A))


def polynomial_to_text(coeffs: np.ndarray, var: str = "λ") -> str:
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = deg - i
        c = float(np.real(c))
        if abs(c) < 1e-10:
            continue
        mag = format_number(abs(c))
        if p == 0:
            body = mag
        else:
            body = ("" if mag == "1" else mag) + var + (f"^{p}" if p > 1 else "")
        sign = "-" if c < 0 else "+"
        terms.append((sign, body))
    if not terms:
        return "0"
    text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, body in terms[1:]:
        text += f" {sign} {body}"
    return text


# ---------------------------------------------------------------------------
# Row reduction
# ---------------------------------------------------------------------------

@dataclass
class RowStep:
    description: str
    matrix: np.ndarray
    pivot: tuple[int, int] | None = None


def rref_steps(M: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, list[RowStep], list[int]]:
    """Gauss–Jordan elimination recording every elementary row operation."""
    A = np.array(M, dtype=float)
    rows, cols = A.shape
    steps = [RowStep("Start", A.copy())]
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot_row = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[pivot_row, c]) < tol:
            continue
        if pivot_row != r:
            A[[r, pivot_row]] = A[[pivot_row, r]]
            steps.append(RowStep(f"Swap R{r + 1} ↔ R{pivot_row + 1}", A.copy(), (r, c)))
        p = A[r, c]
        if abs(p - 1) > tol:
            A[r] = A[r] / p
            steps.append(RowStep(f"R{r + 1} ← R{r + 1} ÷ {format_number(p)}", A.copy(), (r, c)))
        for i in range(rows):
            if i != r and abs(A[i, c]) > tol:
                f = A[i, c]
                A[i] = A[i] - f * A[r]
                op = "−" if f > 0 else "+"
                steps.append(RowStep(f"R{i + 1} ← R{i + 1} {op} {format_number(abs(f))}·R{r + 1}",
                                     A.copy(), (r, c)))
        A[np.abs(A) < tol] = 0.0
        pivots.append(c)
        r += 1
    return A, steps, pivots


# ---------------------------------------------------------------------------
# Linear systems
# ---------------------------------------------------------------------------

@dataclass
class SystemSolution:
    kind: str  # "unique", "infinite", "none"
    particular: np.ndarray | None
    null_basis: np.ndarray  # columns
    least_squares: np.ndarray
    residual: float


def solve_system(A: np.ndarray, b: np.ndarray) -> SystemSolution:
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = float(np.linalg.norm(A @ x_ls - b))
    rA = rank(A)
    rAb = rank(np.column_stack([A, b]))
    N = null_space(A)
    if rA < rAb:
        return SystemSolution("none", None, N, x_ls, residual)
    if N.shape[1] == 0:
        return SystemSolution("unique", x_ls, N, x_ls, residual)
    return SystemSolution("infinite", x_ls, N, x_ls, residual)


# ---------------------------------------------------------------------------
# Orthogonalisation & decompositions
# ---------------------------------------------------------------------------

@dataclass
class GramSchmidtStep:
    index: int
    original: np.ndarray
    projections: list[np.ndarray]  # projection onto each previous q
    orthogonal: np.ndarray  # before normalising
    normalised: np.ndarray | None  # None if the vector was dependent


def gram_schmidt(vectors: np.ndarray) -> list[GramSchmidtStep]:
    """Classical Gram–Schmidt on the columns of ``vectors``, step by step."""
    qs: list[np.ndarray] = []
    steps: list[GramSchmidtStep] = []
    for k in range(vectors.shape[1]):
        v = vectors[:, k].astype(float)
        projections = [np.dot(v, q) * q for q in qs]
        w = v - sum(projections, np.zeros_like(v))
        norm = np.linalg.norm(w)
        q = w / norm if norm > 1e-9 else None
        if q is not None:
            qs.append(q)
        steps.append(GramSchmidtStep(k, v, projections, w, q))
    return steps


def lu_decomposition(A: np.ndarray):
    """LU with partial pivoting: returns P, L, U with P A = L U."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            L[[k, pivot], :k] = L[[pivot, k], :k]
        if abs(U[k, k]) < 1e-12:
            continue
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i] -= L[i, k] * U[k]
    L += np.eye(n)
    U[np.abs(U) < 1e-12] = 0
    return P, L, U


def qr_decomposition(A: np.ndarray):
    Q, R = np.linalg.qr(A)
    # Make R's diagonal non-negative so Q matches Gram–Schmidt.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs, (R.T * signs).T


def svd(A: np.ndarray):
    U, s, Vt = np.linalg.svd(A)
    # Keep both rotations proper (det = +1) where possible for nicer animations.
    if A.shape[0] == A.shape[1]:
        if np.linalg.det(U) < 0 and np.linalg.det(Vt) < 0:
            U[:, -1] *= -1
            Vt[-1] *= -1
    return U, s, Vt


def cholesky(A: np.ndarray):
    if not np.allclose(A, A.T):
        return None
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None


def matrix_exp(A: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series."""
    A = np.asarray(A, dtype=float)
    norm = np.linalg.norm(A, ord=np.inf)
    s = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
    X = A / (2 ** s)
    result = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, 30):
        term = term @ X / k
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1e-16:
            break
    for _ in range(s):
        result = result @ result
    return result


def change_of_basis(v_standard: np.ndarray, basis_columns: np.ndarray) -> np.ndarray:
    """Coordinates of a standard-basis vector in the given basis."""
    return np.linalg.solve(basis_columns, v_standard)


def classify_linear_system_2d(A: np.ndarray) -> str:
    """Name of the phase portrait of x' = A x."""
    tr = float(np.trace(A))
    det = float(np.linalg.det(A))
    disc = tr * tr - 4 * det
    eps = 1e-9
    if abs(det) < eps:
        return "Degenerate (a line of fixed points)"
    if det < 0:
        return "Saddle (unstable)"
    if abs(tr) < eps:
        return "Center (closed orbits)"
    stability = "stable" if tr < 0 else "unstable"
    if disc < -eps:
        return f"Spiral ({stability})"
    if abs(disc) <= eps:
        return f"Degenerate / star node ({stability})"
    return f"Node ({stability})"
