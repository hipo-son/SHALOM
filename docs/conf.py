# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SHALOM'
copyright = '2026, Shinwon Son'
author = 'Shinwon Son'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx_copybutton',
    'nbsphinx',
]

# Markdown file support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# nbsphinx: don't execute notebooks during build
nbsphinx_execute = 'never'

# autodoc: mock imports that may not be available at doc build time
autodoc_mock_imports = [
    'mcp',
    'mp_api',
    'pymatgen',
    'rich',
    'openai',
    'anthropic',
    'spglib',
    'phonopy',
    'matplotlib',
    'seekpath',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Suppress duplicate object warnings (shalom.backends re-exports sub-module classes)
suppress_warnings = ['autodoc.duplicate_object']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
