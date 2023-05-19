# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'PyBCI'
copyright = '2023, Liam Booth'
author = 'Liam Booth'

# The full version, including alpha/beta/rc tags
release = '0.1.4b2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'  # for Sphinx < 2.0
latex_logo = 'Images/pyBCI.png.png'
# -- Options for HTML output -------------------------------------------------

html_theme_options = {
#    'logo': 'logo.png',
    'github_user': 'LMBooth',
    'github_repo': 'PyBCI',
    'github_button': 'true',
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
