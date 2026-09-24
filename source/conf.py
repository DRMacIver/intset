# This file is part of intset (https://github.com/DRMacIver/intset)

# Most of this work is copyright (C) 2013-2026 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others, who hold
# copyright over their individual contributions.

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

import intset

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
project = "IntSet"
copyright = "2015-2026, David R. MacIver"
author = "David R. MacIver"
version = intset.__version__
release = intset.__version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = False
html_theme = "alabaster"
html_static_path = ["_static"]
htmlhelp_basename = "IntSetdoc"
latex_elements = {}
latex_documents = [
    (master_doc, "IntSet.tex", "IntSet Documentation", "David R. MacIver", "manual"),
]
man_pages = [(master_doc, "intset", "IntSet Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "IntSet",
        "IntSet Documentation",
        author,
        "IntSet",
        "Efficient representations of large sets of integers.",
        "Miscellaneous",
    ),
]
autodoc_member_order = "bysource"
