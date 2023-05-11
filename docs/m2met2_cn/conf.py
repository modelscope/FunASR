import guzzle_sphinx_theme
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '多通道多方会议转录挑战2.0'
copyright = '2023, Speech Lab, Alibaba Group; ASLP Group, Northwestern Polytechnical University'
author = 'Speech Lab, Alibaba Group; Audio, Speech and Language Processing Group, Northwestern Polytechnical University'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'myst_parser',
    'guzzle_sphinx_theme'
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_heading_anchors = 2
myst_highlight_code_blocks=True
myst_update_mathjax=False
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'
html_sidebars = {
    '**': ['logo-text.html',
            'searchbox.html',
            'globaltoc.html']
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'
html_static_path = ['_static']
