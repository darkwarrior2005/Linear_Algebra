"""Smoke tests: every page renders without an exception, in 2D and in 3D."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from linalg_viz.registry import SECTIONS  # noqa: E402

MODULES = [spec[0] for specs in SECTIONS.values() for spec in specs]


def _run(module: str, space: str | None = None) -> AppTest:
    at = AppTest.from_string(f"from linalg_viz.pages import {module}\n{module}.render()", default_timeout=180)
    at.run()
    if space:
        for radio in at.radio:
            if radio.label == "Space" and space in radio.options:
                radio.set_value(space)
        at.run()
    return at


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("space", [None, "3D"])
def test_page_renders(module, space):
    at = _run(module, space)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("preset_label", ["Field preset"])
def test_every_field_preset_renders(preset_label):
    at = _run("fields")
    box = next(s for s in at.selectbox if s.label == preset_label)
    for option in box.options:
        if option == "(custom)":
            continue
        box.set_value(option)
        at.run()
        assert not at.exception, (option, [e.value for e in at.exception])
        assert not at.error, (option, [e.value for e in at.error])
        box = next(s for s in at.selectbox if s.label == preset_label)


def test_bad_formula_shows_a_friendly_error():
    at = _run("fields")
    at.text_input[0].set_value("import os")
    at.run()
    assert not at.exception
    assert at.error


def _labels(widgets):
    return [(w.label, list(w.options)) for w in widgets]


@pytest.mark.parametrize("module", [m for m in MODULES if m != "home"])
def test_every_choice_renders(module):
    """Flip every radio, selectbox and multiselect option on the page, one at a time."""
    at = _run(module)
    for kind in ("radio", "selectbox"):
        for label, options in _labels(getattr(at, kind)):
            for option in options:
                widget = next((w for w in getattr(at, kind) if w.label == label), None)
                if widget is None or option not in widget.options:
                    continue
                widget.set_value(option)
                at.run()
                assert not at.exception, (module, label, option, [e.value[:500] for e in at.exception])
    if at.multiselect:
        for ms in at.multiselect:
            ms.set_value(list(ms.options))
        at.run()
        assert not at.exception, (module, "multiselect", [e.value[:500] for e in at.exception])
