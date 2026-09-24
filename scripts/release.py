# This file is part of intset (https://github.com/DRMacIver/intset)

# Most of this work is copyright (C) 2013-2026 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others, who hold
# copyright over their individual contributions.

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

"""Helpers for cutting a release. Driven by the justfile and by the release
workflow in .github/workflows/release.yml; not intended to be run by hand.

Usage:
    release.py version              Print the current package version.
    release.py prepare VERSION      Set the version and stamp the changelog.
    release.py notes VERSION        Print the changelog section for VERSION.
"""

import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "src" / "intset" / "version.py"
CHANGELOG = ROOT / "CHANGELOG.rst"

VERSION_PATTERN = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)
RELEASE_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def current_version():
    match = VERSION_PATTERN.search(VERSION_FILE.read_text())
    assert match is not None, f"Could not find __version__ in {VERSION_FILE}"
    return match.group("version")


def changelog_sections():
    """Split the changelog into (heading, body) pairs, in file order.

    A heading is a line followed by a line of dashes."""
    lines = CHANGELOG.read_text().splitlines()
    sections = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[i + 1]):
            sections.append((line, []))
            i += 2
            continue
        if sections:
            sections[-1][1].append(line)
        i += 1
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def notes(version):
    for heading, body in changelog_sections():
        if heading.split(" - ")[0] == version:
            return body
    sys.exit(f"No changelog section found for version {version}")


def prepare(version):
    if not RELEASE_VERSION_PATTERN.match(version):
        sys.exit(f"{version!r} is not of the form MAJOR.MINOR.PATCH")
    old = current_version()
    if tuple(map(int, version.split("."))) <= tuple(map(int, old.split("."))):
        sys.exit(f"New version {version} is not greater than current {old}")

    changelog = CHANGELOG.read_text()
    unreleased = "Unreleased\n----------\n"
    if unreleased not in changelog:
        sys.exit("CHANGELOG.rst has no 'Unreleased' section to release")
    if not notes("Unreleased"):
        sys.exit("The 'Unreleased' changelog section is empty")
    heading = f"{version} - {datetime.date.today().isoformat()}"
    stamped = f"{heading}\n{'-' * len(heading)}\n"
    CHANGELOG.write_text(changelog.replace(unreleased, stamped, 1))

    VERSION_FILE.write_text(
        VERSION_PATTERN.sub(f'__version__ = "{version}"', VERSION_FILE.read_text())
    )
    print(f"Prepared release {version} (was {old})")


def main(argv):
    match argv:
        case ["version"]:
            print(current_version())
        case ["prepare", version]:
            prepare(version)
        case ["notes", version]:
            print(notes(version))
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
