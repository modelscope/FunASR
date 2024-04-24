# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MULTI-PARTY MEETING TRANSCRIPTION CHALLENGE 2.0"
copyright = "2023, Speech Lab, Alibaba Group; ASLP Group, Northwestern Polytechnical University"
author = "Speech Lab, Alibaba Group; Audio, Speech and Language Processing Group, Northwestern Polytechnical University"


extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # "sphinxarg.ext",
    "sphinx_markdown_tables",
    # 'recommonmark',
    "sphinx_rtd_theme",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_heading_anchors = 2
myst_highlight_code_blocks = True
myst_update_mathjax = False

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
