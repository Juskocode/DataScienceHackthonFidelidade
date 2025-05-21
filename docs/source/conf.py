"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../xsell_dental_exemplo"))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "xsell_dental_exemplo"
copyright = "2022, CAA Developers"
author = "CAA Developers"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

master_doc = "index"
autoclass_content = "both"
pygments_style = "sphinx"
autodoc_member_order = "bysource"
html4_writer = True
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx_rtd_theme"
]
intersphinx_mapping = {"numpy": ("https://docs.scipy.org/doc/numpy/", None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
}
html_show_sphinx = False

autoapi_type = 'python'
autoapi_dirs = ['../../xsell_dental_exemplo']
