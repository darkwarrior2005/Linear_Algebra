"""Smoke tests for the single-graph workspace: every layer, option and preset must render."""

import pytest

pytest.importorskip("streamlit")
from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(dim: str = "2D") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    if dim == "3D":
        at.radio(key="ws_dim").set_value("3D")
        at.run()
    return at


def _ok(at: AppTest, what="") -> None:
    assert not at.exception, (what, [e.value[:800] for e in at.exception])
    assert not at.error, (what, [e.value for e in at.error])


def _everything_on(at: AppTest) -> None:
    for t in at.toggle:
        t.set_value(True)
    at.run()
    for g in at.get("button_group"):
        if g.label != "▶ Play animates":
            g.set_value(list(g.options))
    at.checkbox(key="ws_then_b").set_value(True)
    at.checkbox(key="ws_undo").set_value(True)
    at.checkbox(key="ws_fp").set_value(True)
    at.run()


@pytest.mark.parametrize("dim", ["2D", "3D"])
def test_default_renders(dim):
    _ok(_app(dim))


@pytest.mark.parametrize("dim", ["2D", "3D"])
@pytest.mark.parametrize("play", ["the matrix", "time"])
def test_all_layers_together(dim, play):
    at = _app(dim)
    _everything_on(at)
    _ok(at, "all on")
    at.get("button_group")[0].set_value(play)
    at.checkbox(key="ws_flow").set_value(True)
    at.run()
    _ok(at, play)


@pytest.mark.parametrize("dim", ["2D", "3D"])
def test_every_preset_and_choice(dim):
    at = _app(dim)
    _everything_on(at)
    for box in list(at.selectbox):
        label, key = box.label, box.key
        for option in box.options:
            widget = at.selectbox(key=key)
            if option not in widget.options:
                continue
            widget.set_value(option)
            at.run()
            _ok(at, (label, option))
    for radio in list(at.radio):
        for option in radio.options:
            at.radio(key=radio.key).set_value(option)
            at.run()
            _ok(at, (radio.label, option))


def test_play_modes_and_matrix_field():
    at = _app()
    at.toggle(key="ws_field_on").set_value(True)
    at.radio(key="ws_field_src").set_value("the matrix: F = A·x")
    at.run()
    _ok(at)
    for play in ["auto", "the matrix", "time"]:
        at.get("button_group")[0].set_value(play)
        at.run()
        _ok(at, play)


def test_bad_formula_shows_a_friendly_error():
    at = _app()
    at.toggle(key="ws_field_on").set_value(True)
    at.run()
    at.text_input(key="ws_field2_0").set_value("import os")
    at.run()
    assert not at.exception
    assert at.error


def test_singular_undo_warns_instead_of_crashing():
    at = _app()
    at.selectbox(key="ws_A_2x2__preset").set_value("Project onto x-axis")
    at.checkbox(key="ws_undo").set_value(True)
    at.run()
    assert not at.exception
    assert any("no inverse" in w.value for w in at.warning)


def test_calculator_sends_matrix_to_graph():
    at = _app()
    at.text_area(key="calc_A").set_value("0 -1\n1 0")
    at.run()
    at.button(key="calc_send").click()
    at.run()
    _ok(at)
    assert at.text_input(key="ws_A_2x2__0_1").value == "-1"
