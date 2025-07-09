# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# set of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import jax.typing


sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "AstroDynX"
copyright = "2025"
author = "AstroDynX contributors"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx_math_dollar",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for autodoc -----------------------------------------------------
_TYPE_MAP = {
    jax.typing.ArrayLike: ":py:data:`~jax.typing.ArrayLike`",
    jax.typing.DTypeLike: ":py:data:`~jax.typing.DTypeLike`",
}


def custom_typehints_formatter(annotation: str, config) -> str | None:
    return _TYPE_MAP.get(annotation)


typehints_formatter = custom_typehints_formatter

always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "diffrax": ("https://docs.kidger.site/diffrax/", None),
}

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True

# -- Options for MyST --------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/adxorg/astrodynx",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
}
html_static_path = ["_static"]
html_css_files = [
    "custom-api.css",
]

# -- Options for LaTeX output ------------------------------------------------
latex_engine = "pdflatex"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}
