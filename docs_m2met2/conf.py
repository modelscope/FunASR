# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'm2met2'
copyright = '2023, Speech Lab, Alibaba Group; Audio, Speech and Language Processing Group, Northwestern Polytechnical University'
author = 'Speech Lab, Alibaba Group; Audio, Speech and Language Processing Group, Northwestern Polytechnical University'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx_rtd_theme',
    "sphinx.ext.autodoc",
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_markdown_tables",
    "sphinx.ext.githubpages",
    'recommonmark',
]
source_suffix = [".rst", ".md"]
templates_path = ['_templates']
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_patterns = []
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
