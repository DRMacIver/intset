# This file is part of intset (https://github.com/DRMacIver/intset)

# Most of this work is copyright (C) 2013-2026 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others, who hold
# copyright over their individual contributions.

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

import doctest
from pathlib import Path

README = Path(__file__).parent.parent / "README.rst"


def test_readme_examples_are_correct():
    result = doctest.testfile(str(README), module_relative=False)
    assert result.attempted > 0
    assert result.failed == 0
