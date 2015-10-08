import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'IntSet'
copyright = '2015, David R. MacIver'
author = 'David R. MacIver'
version = '1.0.0'
release = '1.0.0'
language = None
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'alabaster'
html_static_path = ['_static']
htmlhelp_basename = 'IntSetdoc'
latex_elements = {
}
latex_documents = [
    (master_doc, 'IntSet.tex', 'IntSet Documentation',
     'David R. MacIver', 'manual'),
]
man_pages = [
    (master_doc, 'intset', 'IntSet Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'IntSet', 'IntSet Documentation',
     author, 'IntSet', 'One line description of project.',
     'Miscellaneous'),
]
autodoc_member_order = 'bysource'
