"""Shared Streamlit widgets: matrix/vector editors, plot display, small helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import streamlit as st

from .core.parsing import ExpressionError, eval_number, format_number, matrix_to_latex, vector_to_latex

PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def show(fig, key: str | None = None) -> None:
    """Display a Plotly figure full-width with the app's own (dark) styling."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)
    except TypeError:  # older Streamlit without the ``width`` argument
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOT_CONFIG, key=key)


def header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def dimension_picker(key: str, default: int = 2, label: str = "Space") -> int:
    choice = st.radio(label, ["2D", "3D"], index=0 if default == 2 else 1, horizontal=True, key=key)
    return 2 if choice == "2D" else 3


def _cell_key(key: str, i: int, j: int) -> str:
    return f"{key}__{i}_{j}"


def _set_cells(key: str, M: Sequence[Sequence], rows: int, cols: int) -> None:
    for i in range(rows):
        for j in range(cols):
            st.session_state[_cell_key(key, i, j)] = str(M[i][j])


def matrix_editor(key: str, rows: int, cols: int, default: Sequence[Sequence],
                  presets: Mapping[str, Sequence[Sequence]] | None = None, label: str = "Matrix",
                  show_latex: bool = True, help_text: str | None = None) -> np.ndarray | None:
    """Grid of small text boxes; each cell accepts numbers or expressions like ``cos(pi/4)``."""
    key = f"{key}_{rows}x{cols}"
    if presets:
        names = ["(custom)"] + list(presets)

        def apply_preset() -> None:
            name = st.session_state[f"{key}__preset"]
            if name in presets:
                _set_cells(key, presets[name], rows, cols)

        st.selectbox(f"{label} preset", names, key=f"{key}__preset", on_change=apply_preset,
                     help="Pick a classic transformation, then edit any entry.")
    else:
        st.markdown(f"**{label}**")
    for i in range(rows):
        for j in range(cols):
            ck = _cell_key(key, i, j)
            if ck not in st.session_state:
                st.session_state[ck] = str(default[i][j])
    M = np.zeros((rows, cols))
    errors = []
    for i in range(rows):
        cells = st.columns(cols)
        for j in range(cols):
            text = cells[j].text_input(f"{label}[{i + 1},{j + 1}]", key=_cell_key(key, i, j),
                                       label_visibility="collapsed", help=help_text)
            try:
                M[i, j] = eval_number(text)
            except (ExpressionError, TypeError, ValueError) as exc:
                errors.append(f"Entry ({i + 1},{j + 1}): {exc}")
    if errors:
        st.error("\n\n".join(errors))
        return None
    if show_latex:
        st.latex(matrix_to_latex(M))
    return M


def vector_editor(key: str, dim: int, default: Sequence, label: str = "v",
                  color: str | None = None) -> np.ndarray | None:
    key = f"{key}_{dim}"
    cols = st.columns([1] + [2] * dim)
    swatch = f"<span style='color:{color}'>■</span> " if color else ""
    cols[0].markdown(f"{swatch}**{label}**", unsafe_allow_html=True)
    v = np.zeros(dim)
    for j in range(dim):
        ck = _cell_key(key, 0, j)
        if ck not in st.session_state:
            st.session_state[ck] = str(default[j])
        text = cols[j + 1].text_input(f"{label}[{j + 1}]", key=ck, label_visibility="collapsed")
        try:
            v[j] = eval_number(text)
        except (ExpressionError, TypeError, ValueError) as exc:
            st.error(f"{label}: {exc}")
            return None
    return v


def latex_matrix(M: np.ndarray, prefix: str = "", digits: int = 3) -> None:
    st.latex(prefix + matrix_to_latex(M, digits))


def latex_vector(v: np.ndarray, prefix: str = "", digits: int = 3) -> None:
    st.latex(prefix + vector_to_latex(v, digits))


def fmt(v, digits: int = 3) -> str:
    return format_number(v, digits)


def fmt_vec(v, digits: int = 3) -> str:
    return "(" + ", ".join(format_number(x, digits) for x in np.asarray(v).ravel()) + ")"


def tip(text: str) -> None:
    st.info(text, icon="💡")


def set_matrix(key: str, M: np.ndarray, digits: int = 4) -> None:
    """Overwrite the cells of a :func:`matrix_editor` (call from a button callback)."""
    M = np.atleast_2d(M)
    rows, cols = M.shape
    _set_cells(f"{key}_{rows}x{cols}", [[format_number(v, digits) for v in row] for row in M], rows, cols)


def table(rows) -> None:
    try:
        st.dataframe(rows, hide_index=True, width="stretch")
    except TypeError:  # older Streamlit
        st.dataframe(rows, hide_index=True, use_container_width=True)
