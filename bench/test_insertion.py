# coding=utf-8

# This file is part of intset (https://github.com/DRMacIver/inteset)

# Most of this work is copyright (C) 2013-2015 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others, who hold
# copyright over their individual contributions.

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

from random import Random
import hashlib
import pytest
from intset import IntSet
import sys
import unicodedata


if sys.version_info[0] == 3:
    unichr = chr


SIZE = 10000

INSERTIONS = {
    "range": list(range(SIZE)),
}

for i in [2, 5, 9]:
    INSERTIONS["gaps(%d)" % (i,)] = list(range(0, SIZE * i, i))


for k, v in list(INSERTIONS.items()):
    INSERTIONS["%s-reversed" % (k,)] = list(reversed(v))
    for i in [1, 2]:
        rnd = Random(hashlib.sha1(str(i)).digest())
        t = list(v)
        rnd.shuffle(t)
        INSERTIONS["%s-shuffled(%d)" % (k, i)] = t


INSERTIONS = list(sorted(INSERTIONS.items()))


@pytest.mark.parametrize(
    'dataset', [v for _, v in INSERTIONS], ids=[k for k, _ in INSERTIONS]
)
def test_set_insertion_one_by_one(dataset, benchmark):
    @benchmark
    def result():
        d = set()
        for v in dataset:
            d.add(v)
        return d
    assert list(result) == sorted(dataset)


@pytest.mark.parametrize(
    'dataset', [v for _, v in INSERTIONS], ids=[k for k, _ in INSERTIONS]
)
def test_insertion_one_by_one(dataset, benchmark):
    @benchmark
    def result():
        d = IntSet.empty()
        for v in dataset:
            d = d.insert(v)
        return d
    assert list(result) == sorted(dataset)


@pytest.mark.parametrize(
    'dataset', [v for _, v in INSERTIONS], ids=[k for k, _ in INSERTIONS]
)
def test_insertion_with_builder(dataset, benchmark):
    @benchmark
    def result():
        d = IntSet.Builder()
        for v in dataset:
            d.insert(v)
        return d.build()
    assert list(result) == sorted(dataset)


def test_build_unicode_categories(benchmark):
    @benchmark
    def result():
        x = {}
        for i in range(sys.maxunicode + 1):
            category = unicodedata.category(unichr(i))
            if category not in x:
                x[category] = IntSet.Builder()
            x[category].insert(i)
        for k, v in list(x.items()):
            x[k] = v.build()
        return x
    assert sum(len(v) for v in result.values()) == sys.maxunicode + 1
