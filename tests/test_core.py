import numpy as np
import pytest

from linalg_viz.core import linalg as la
from linalg_viz.core.fields import ParametricCurve, ScalarField, VectorField, find_fixed_points, rk4_paths, field_rhs
from linalg_viz.core.parsing import ExpressionError, compile_expr, eval_number, parse_matrix, parse_vector


# --------------------------------------------------------------------- parsing

def test_numbers_and_expressions():
    assert eval_number("sqrt(2)/2") == pytest.approx(np.sqrt(2) / 2)
    assert eval_number("2pi") == pytest.approx(2 * np.pi)
    assert eval_number("1e-3") == pytest.approx(1e-3)
    assert eval_number("2^3") == 8


def test_matrix_and_vector_parsing():
    M = parse_matrix("1 2\n3 4")
    assert M.tolist() == [[1, 2], [3, 4]]
    M = parse_matrix("cos(pi/3), -sin(pi/3); sin(pi/3), cos(pi/3)")
    assert np.allclose(M.T @ M, np.eye(2))
    assert parse_vector("[atan2(1, 1), 2]") == pytest.approx([np.pi / 4, 2])
    with pytest.raises(ExpressionError):
        parse_matrix("1 2\n3")


@pytest.mark.parametrize("bad", ["__import__('os')", "x.real", "open('f')", "(lambda: 1)()", "[1, 2]"])
def test_unsafe_input_is_rejected(bad):
    with pytest.raises(ExpressionError):
        compile_expr(bad, ("x",))


def test_expression_broadcasts_constants():
    f = compile_expr("3", ("x", "y"))
    assert f(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)
    g = compile_expr("2x + y^2", ("x", "y"))
    assert g(np.array([1.0]), np.array([3.0]))[0] == 11


# ---------------------------------------------------------------------- linalg

def test_rank_and_subspaces():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
    assert la.rank(A) == 2
    N = la.null_space(A)
    assert N.shape == (3, 1) and np.allclose(A @ N, 0)
    assert la.column_space(A).shape == (3, 2)


def test_eigen_analysis_real_complex_and_defective():
    pairs = la.eigen_analysis(np.array([[3, 1], [0, 2]], float))
    assert sorted(p.value.real for p in pairs) == pytest.approx([2, 3])
    for p in pairs:
        A = np.array([[3, 1], [0, 2]], float)
        assert np.allclose(A @ p.vector, p.value * p.vector)
    rot = la.eigen_analysis(np.array([[0, -1], [1, 0]], float))
    assert all(not p.is_real for p in rot)
    shear = np.array([[1, 1], [0, 1]], float)
    assert not la.is_diagonalizable(shear)
    assert la.is_diagonalizable(np.eye(2))


def test_rref_and_solve():
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], float)
    b = np.array([8, -11, -3], float)
    R, steps, pivots = la.rref_steps(np.column_stack([A, b]))
    assert np.allclose(R[:, -1], [2, 3, -1]) and pivots == [0, 1, 2]
    assert la.solve_system(A, b).kind == "unique"
    assert la.solve_system(np.array([[1, 1], [2, 2]], float), np.array([1, 3], float)).kind == "none"
    sol = la.solve_system(np.array([[1, 1], [2, 2]], float), np.array([1, 2], float))
    assert sol.kind == "infinite" and sol.null_basis.shape == (2, 1)


def test_decompositions_reconstruct():
    A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], float)
    P, L, U = la.lu_decomposition(A)
    assert np.allclose(P @ A, L @ U)
    Q, R = la.qr_decomposition(A)
    assert np.allclose(Q @ R, A) and np.all(np.diag(R) >= 0)
    U, s, Vt = la.svd(A)
    assert np.allclose(U @ np.diag(s) @ Vt, A)
    steps = la.gram_schmidt(A)
    Qgs = np.column_stack([s.normalised for s in steps])
    assert np.allclose(Qgs.T @ Qgs, np.eye(3))


def test_matrix_exponential():
    A = np.array([[0, -1], [1, 0]], float)
    E = la.matrix_exp(A * np.pi / 2)
    assert np.allclose(E, [[0, -1], [1, 0]], atol=1e-10)
    B = np.array([[-2.0, 7.0], [0.3, 5.0]])
    w, V = np.linalg.eig(B)
    assert np.allclose(la.matrix_exp(B), (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real)


def test_classification():
    assert la.classify_linear_system_2d(np.array([[1, 0], [0, -1]], float)).startswith("Saddle")
    assert la.classify_linear_system_2d(np.array([[0, 1], [-1, 0]], float)).startswith("Center")
    assert la.classify_linear_system_2d(np.array([[-0.3, -1], [1, -0.3]], float)) == "Spiral (stable)"


# ---------------------------------------------------------------------- fields

def test_divergence_and_curl():
    F = VectorField(["-y", "x"])
    assert F.divergence_at([0.3, -0.2]) == pytest.approx(0, abs=1e-6)
    assert F.curl_at([0.3, -0.2]) == pytest.approx(2, abs=1e-6)
    G = VectorField(["x*y", "y*z", "z*x"])
    assert G.divergence_at([1, 2, 3]) == pytest.approx(2 + 3 + 1, abs=1e-5)
    assert G.curl_at([1, 2, 3]) == pytest.approx([-2, -3, -1], abs=1e-5)
    sym = G.symbolic()
    assert str(sym["div"]) == "x + y + z"


def test_scalar_gradient_and_laplacian():
    f = ScalarField("x^2 + 3*y^2", 2)
    X, Y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    G = f.gradient_grid([X, Y])
    assert np.allclose(G[0], 2 * X, atol=1e-6) and np.allclose(G[1], 6 * Y, atol=1e-6)
    assert np.allclose(f.laplacian_grid([X, Y]), 8, atol=1e-3)


def test_rk4_follows_a_circle_and_time_dependence():
    F = VectorField(["-y", "x"])
    _, paths = rk4_paths(field_rhs(F), np.array([[1.0, 0.0]]), 0, 2 * np.pi, 400)
    assert np.allclose(paths[-1, 0], [1, 0], atol=1e-6)
    H = VectorField(["cos(t)", "0"])
    assert H.time_dependent and not F.time_dependent
    _, p = rk4_paths(field_rhs(H), np.zeros((1, 2)), 0, np.pi / 2, 200)
    assert p[-1, 0, 0] == pytest.approx(1, abs=1e-6)


def test_fixed_points():
    pts = find_fixed_points(VectorField(["y", "-sin(x)"]), (-4, 4, -4, 4))
    xs = sorted(round(p.point[0], 3) for p in pts)
    assert xs == pytest.approx([-np.pi, 0, np.pi], abs=1e-3)
    kinds = {round(p.point[0], 3): p.kind for p in pts}
    assert kinds[0.0].startswith("Center") and kinds[round(np.pi, 3)].startswith("Saddle")


def test_curve_kinematics_helix():
    c = ParametricCurve(["cos(t)", "sin(t)", "t"])
    k = c.kinematics(np.array([0.7]))
    # helix x=cos t, y=sin t, z=t: curvature = torsion = 1/2, speed = sqrt 2
    assert k["speed"][0] == pytest.approx(np.sqrt(2), rel=1e-6)
    assert k["curvature"][0] == pytest.approx(0.5, rel=1e-4)
    assert k["torsion"][0] == pytest.approx(0.5, rel=1e-3)
    assert np.dot(k["T"][0], k["N"][0]) == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------- animation paths

@pytest.mark.parametrize("A", [
    [[0, -1], [1, 0]],                                   # 90° rotation
    [[-1, 0], [0, -1]],                                  # 180° rotation (no real log)
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],                   # 120° turn about (1,1,1)
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],                 # 180° about x
])
def test_rotations_turn_without_shrinking(A):
    A = np.array(A, float)
    path = la.transform_path(A)
    assert np.allclose(path(0), np.eye(len(A))) and np.allclose(path(1), A, atol=1e-8)
    for t in np.linspace(0, 1, 11):
        M = path(t)
        assert np.allclose(M.T @ M, np.eye(len(A)), atol=1e-8)  # every frame is a rigid rotation


def test_paths_for_shear_reflection_projection():
    shear = np.array([[1, 1], [0, 1]], float)
    assert np.allclose(la.transform_path(shear)(0.5), [[1, 0.5], [0, 1]])
    for A in ([[0, 1], [1, 0]], [[1, 0], [0, 0]], [[1, -1], [1, 1]], [[1, 2], [0.5, 1]]):
        A = np.array(A, float)
        path = la.transform_path(A)
        assert np.allclose(path(0), np.eye(2)) and np.allclose(path(1), A, atol=1e-8)


def test_real_log():
    A = np.array([[2, 1], [1, 2]], float)
    assert np.allclose(la.matrix_exp(la.real_log(A)), A)
    assert la.real_log(np.array([[1, 0], [0, -1]], float)) is None
