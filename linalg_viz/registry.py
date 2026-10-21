"""Page registry: the sidebar layout, shared by app.py and the welcome page's links."""

from __future__ import annotations

import importlib

import streamlit as st

# (module, title, icon, url) grouped into sidebar sections
SECTIONS = {
    "Start": [
        ("home", "Welcome", "🏠", "home"),
    ],
    "Vectors & matrices": [
        ("vectors", "Vectors", "➡️", "vectors"),
        ("transform", "Linear transformations", "🔀", "transformations"),
        ("compose", "Composition & inverse", "🔗", "composition"),
        ("systems", "Systems  Ax = b", "🎯", "systems"),
        ("subspaces", "Subspaces & rank", "🧊", "subspaces"),
        ("basis", "Change of basis", "🧭", "change-of-basis"),
        ("orthogonal", "Gram–Schmidt & projections", "📏", "gram-schmidt"),
    ],
    "Eigen & decompositions": [
        ("eigen", "Eigenvectors & eigenvalues", "🌟", "eigen"),
        ("decompositions", "Decompositions (SVD, QR, LU…)", "🧩", "decompositions"),
        ("calculator", "Matrix calculator", "🧮", "calculator"),
    ],
    "Fields & time evolution": [
        ("fields", "Vector fields", "🌬️", "vector-fields"),
        ("operators", "Differential operators", "🧲", "operators"),
        ("flows", "Flows & particles", "🌊", "flows"),
        ("dynamics", "Linear dynamics  x′ = Ax", "🌀", "linear-dynamics"),
        ("motion", "Motion along a curve  r(t)", "🛰️", "motion"),
    ],
}

# url -> st.Page, filled by build_pages() on every run
PAGES: dict[str, st.Page] = {}


def _page(module: str, title: str, icon: str, url: str, default: bool) -> st.Page:
    def run() -> None:
        importlib.import_module(f"linalg_viz.pages.{module}").render()

    run.__name__ = f"page_{module}"
    return st.Page(run, title=title, icon=icon, url_path=url, default=default)


def build_pages() -> dict[str, list[st.Page]]:
    sections = {}
    for section, specs in SECTIONS.items():
        pages = []
        for i, (module, title, icon, url) in enumerate(specs):
            page = _page(module, title, icon, url, default=(section == "Start" and i == 0))
            PAGES[url] = page
            pages.append(page)
        sections[section] = pages
    return sections
