# coding=utf-8

# This file is part of intset (https://github.com/DRMacIver/intset)

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

import os

from setuptools import find_packages, setup


def local_file(name):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), name))

SOURCE = local_file("src")
README = local_file("README.rst")


# Assignment to placate pyflakes. The actual version is from the exec that
# follows.
__version__ = None

with open(local_file("src/intset/version.py")) as o:
    exec(o.read())

assert __version__ is not None

setup(
    name='intset',
    version=__version__,
    author='David R. MacIver',
    author_email='david@drmaciver.com',
    packages=find_packages(SOURCE),
    package_dir={"": SOURCE},
    url='https://github.com/DRMacIver/intset',
    license='MPL v2',
    description='A library for property based testing',
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Testing",
    ],
    long_description=open(README).read(),
)
