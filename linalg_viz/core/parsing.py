"""Safe parsing of user input: numbers, vectors, matrices and math expressions.

Everything a user types goes through an AST whitelist before it is evaluated,
so expressions such as ``sin(x)*exp(-t)`` work while arbitrary Python
(``__import__('os')``) is rejected.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# Functions a user may call inside an expression, mapped to numpy versions.
NUMPY_FUNCS: dict[str, Callable] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "atan2": np.arctan2, "arctan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil,
    "min": np.minimum, "max": np.maximum,
    "where": np.where, "heaviside": lambda a: np.heaviside(a, 0.5),
    "sec": lambda a: 1 / np.cos(a), "csc": lambda a: 1 / np.sin(a), "cot": lambda a: 1 / np.tan(a),
}

CONSTANTS: dict[str, float] = {"pi": np.pi, "e": np.e, "tau": 2 * np.pi, "inf": np.inf}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


class ExpressionError(ValueError):
    """Raised when user input cannot be understood."""


def _normalise(text: str) -> str:
    text = text.strip().replace("^", "**").replace("×", "*").replace("−", "-")
    # Allow implicit multiplication between a number and a name: 2x -> 2*x, 3pi -> 3*pi
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d*)?)\s*(?![eE][-+]?\d)([a-zA-Z_(])", r"\1*\2", text)
    # ")(" and ")x" -> ")*(" and ")*x"
    text = re.sub(r"\)\s*([a-zA-Z_(\d])", r")*\1", text)
    return text


def _validate(tree: ast.AST, variables: Iterable[str]) -> None:
    allowed_names = set(variables) | set(NUMPY_FUNCS) | set(CONSTANTS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"'{type(node).__name__}' is not allowed in an expression")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ExpressionError(
                f"Unknown name '{node.id}'. Use variables {sorted(variables)} "
                f"or functions like sin, cos, exp, sqrt"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in NUMPY_FUNCS:
                raise ExpressionError("Only math functions such as sin(), exp(), sqrt() may be called")
            if node.keywords:
                raise ExpressionError("Keyword arguments are not supported")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex)):
            raise ExpressionError("Only numbers are allowed as constants")


@dataclass
class Expression:
    """A compiled, validated math expression in a fixed set of variables."""

    text: str
    variables: tuple[str, ...]
    _code: object = field(repr=False, default=None)

    def __post_init__(self) -> None:
        source = _normalise(self.text)
        if not source:
            raise ExpressionError("Empty expression")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Could not read '{self.text}': {exc.msg}") from None
        _validate(tree, self.variables)
        self.source = source
        self._code = compile(tree, "<expression>", "eval")
        self.uses = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & set(self.variables)

    def __call__(self, *args, **kwargs):
        env = dict(zip(self.variables, args))
        env.update(kwargs)
        namespace = {"__builtins__": {}, **NUMPY_FUNCS, **CONSTANTS, **env}
        with np.errstate(all="ignore"):
            value = eval(self._code, namespace)  # noqa: S307 - validated AST, no builtins
        arrays = [np.asarray(a, dtype=float) for a in env.values()]
        shape = np.broadcast_shapes(*[a.shape for a in arrays]) if arrays else ()
        value = np.asarray(value, dtype=float)
        if value.shape != shape:
            value = np.broadcast_to(value, shape).copy()
        return value

    def to_sympy(self):
        """Return a SymPy expression, or ``None`` if it has no symbolic form."""
        try:
            import sympy as sp
        except ImportError:  # pragma: no cover - sympy is optional
            return None
        local = {v: sp.Symbol(v, real=True) for v in self.variables}
        local.update({
            "ln": sp.log, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
            "atan2": sp.atan2, "arctan2": sp.atan2, "abs": sp.Abs, "sign": sp.sign,
            "log10": lambda a: sp.log(a, 10), "log2": lambda a: sp.log(a, 2),
            "min": sp.Min, "max": sp.Max, "heaviside": sp.Heaviside,
            "pi": sp.pi, "e": sp.E, "tau": 2 * sp.pi, "inf": sp.oo,
            "sec": sp.sec, "csc": sp.csc, "cot": sp.cot, "floor": sp.floor, "ceil": sp.ceiling,
        })
        if "where" in self.source:
            return None
        try:
            return sp.sympify(self.source, locals=local)
        except Exception:  # noqa: BLE001 - any sympy failure means "no symbolic form"
            return None


def compile_expr(text: str, variables: Sequence[str]) -> Expression:
    return Expression(str(text), tuple(variables))


def eval_number(text: str) -> float:
    """Evaluate a constant expression such as ``'sqrt(2)/2'`` or ``'-pi/4'``."""
    value = compile_expr(text, ())()
    return float(value)


_SPLIT_ROWS = re.compile(r"[;\n]+")
_SPLIT_ENTRIES = re.compile(r"[,\s]+")


def _split_entries(row: str) -> list[str]:
    """Split a row on commas/whitespace, but keep things like 'atan2(1, 2)' intact."""
    tokens, depth, current = [], 0, ""
    for ch in row:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and (ch == "," or ch.isspace()):
            if current.strip():
                tokens.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())
    return tokens


def parse_vector(text: str) -> np.ndarray:
    """Parse ``'1, 2, 3'`` / ``'[1 2 3]'`` / ``'(cos(pi/3), 1)'`` into an array."""
    cleaned = text.strip().strip("[]()").strip()
    if not cleaned:
        raise ExpressionError("Empty vector")
    return np.array([eval_number(tok) for tok in _split_entries(cleaned)], dtype=float)


def parse_matrix(text: str) -> np.ndarray:
    """Parse a matrix written one row per line (or rows separated by ';')."""
    cleaned = text.strip().replace("[", " ").replace("]", " ").strip()
    rows = [r for r in _SPLIT_ROWS.split(cleaned) if r.strip()]
    if not rows:
        raise ExpressionError("Empty matrix")
    data = [[eval_number(tok) for tok in _split_entries(r)] for r in rows]
    width = len(data[0])
    if any(len(r) != width for r in data):
        raise ExpressionError("Every row of the matrix must have the same number of entries")
    return np.array(data, dtype=float)


def format_number(value: float, digits: int = 3) -> str:
    """Pretty-print a float: integers without decimals, tiny values as 0."""
    if isinstance(value, complex) or np.iscomplexobj(value):
        value = complex(value)
        if abs(value.imag) < 10 ** -(digits + 2):
            return format_number(value.real, digits)
        re_part = format_number(value.real, digits)
        im_part = format_number(abs(value.imag), digits)
        sign = "+" if value.imag >= 0 else "−"
        return f"{re_part} {sign} {im_part}i" if abs(value.real) > 10 ** -(digits + 2) else (
            f"{'-' if value.imag < 0 else ''}{im_part}i")
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if abs(value) < 10 ** -(digits + 2):
        return "0"
    if abs(value - round(value)) < 10 ** -(digits + 2):
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def matrix_to_text(M: np.ndarray, digits: int = 3) -> str:
    M = np.atleast_2d(M)
    return "\n".join("  ".join(format_number(v, digits) for v in row) for row in M)


def matrix_to_latex(M: np.ndarray, digits: int = 3) -> str:
    M = np.atleast_2d(np.asarray(M))
    body = r" \\ ".join(" & ".join(format_number(v, digits) for v in row) for row in M)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"


def vector_to_latex(v: np.ndarray, digits: int = 3) -> str:
    return matrix_to_latex(np.asarray(v).reshape(-1, 1), digits)
