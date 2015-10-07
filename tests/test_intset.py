# coding=utf-8


# This file is part of Hypothesis (https://github.com/DRMacIver/hypothesis)

# Most of this work is copyright (C) 2013-2015 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# https://github.com/DRMacIver/hypothesis/blob/master/CONTRIBUTING.rst for a
# full list of people who may hold copyright, and consult the git log if you
# need to determine who owns an individual contribution.

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

import operator as op
import os

import hypothesis.strategies as st
import pytest
from hypothesis import assume, example, given
from hypothesis.stateful import Bundle, rule, RuleBasedStateMachine
from intset import IntSet
from intset.intset import Interval

if os.getenv('HYPOTHESIS_PROFILE') == 'coverage':
    from hypothesis import Settings
    Settings.default.max_examples = 0


def test_not_equal_to_other_types():
    assert IntSet.single(1) != 1


integers_in_range = st.one_of(
    st.integers(min_value=0, max_value=2 ** 64 - 1),
    st.integers(min_value=0).filter(lambda x: x < 2 ** 64))

intervals = st.tuples(integers_in_range, integers_in_range).map(
    lambda x: sorted(tuple(x)))

SMALL = 100

short_intervals = st.builds(
    lambda start, length: assume(
        start + length <= 2 ** 64) and (start, start + length),
    integers_in_range, st.integers(0, SMALL))

interval_list = st.lists(intervals, average_size=10)

IntSets = st.builds(
    IntSet.from_intervals, interval_list) | integers_in_range.map(
        IntSet.single) | intervals.map(lambda x: IntSet.interval(*x))


@example([
    [0, 1], [13356455824667928169, 13356455824667928173],
    [0, 13356455824667928169]])
@example([
    [7341163011306295800, 7341163011306295801], [0, 7341163011306295799],
    [7341163011306295799, 7341163011306295800]])
@example([
    [5072311219282295777, 5072311219282295778],
    [5072311219282295775, 5072311219282295776], [0, 5072311219282295775]])
@example([[2, 5], [0, 1]])
@given(interval_list)
def test_sequentially_removing_intervals_yields_empty(ls):
    running = IntSet.from_intervals(ls)
    for i in ls:
        inter = IntSet.interval(*i)
        extra = inter & running
        original = running.size()
        assert (running - inter) == (running - extra)
        running -= inter
        assert running.size() == original - extra.size()
    assert running.size() == 0


@example(set())
@example(set([1, 2, 3]))
@given(st.sets(integers_in_range))
def test_instantiating_with_a_collection_gives_that_collection(xs):
    ts = IntSet.from_iterable(xs)
    assert len(ts) == len(xs)
    assert set(ts) == xs


def test_trying_to_pass_extra_args_to_intset_errors():
    with pytest.raises(TypeError):
        IntSet.from_iterable(1, 2)


@example(IntSet.from_iterable([]), 1)
@example(IntSet.from_iterable([]), -1)
@example(IntSet.from_iterable([0]), -2)
@example(IntSet.from_iterable([0]), 2)
@given(IntSets, st.integers())
def test_raises_index_error_out_of_bounds(x, i):
    assume(abs(i) > x.size())
    with pytest.raises(IndexError):
        x[i]


SmallIntSets = st.builds(
    IntSet.from_intervals,
    st.lists(short_intervals,
             average_size=5)).filter(lambda x: x.size() <= SMALL)


@example(IntSet.empty(), IntSet.from_iterable([1]))
@example(IntSet.from_iterable([2]), IntSet.from_iterable([1]))
@example(IntSet.from_iterable([1, 2]), IntSet.from_iterable([1]))
@given(IntSets, IntSets)
def test_ordering_method_consistency(x, y):
    assert (x <= y) == (not (x > y))
    assert (x >= y) == (not (x < y))


def assert_order(x, y):
    assert x < y
    assert y >= x
    assert x != y


@given(integers_in_range)
def test_discarding_a_solo_gives_the_empty_list(i):
    imp = IntSet.single(i)
    assert imp.discard(i).size() == 0


@given(IntSets)
def test_inserting_a_smaller_element_produces_a_smaller_intset(imp):
    assume(imp.size() > 0)
    i = imp[0]
    assume(i > 0)
    impy = imp.insert(i - 1)
    assert_order(impy, imp)


@given(IntSets)
def test_deleting_the_largest_element_produces_a_smaller_intset(imp):
    assume(imp.size() > 1)
    i = imp[-1]
    impy = imp.discard(i)
    assert_order(impy, imp)


@given(IntSets)
@example(IntSet.from_iterable([0, 2]))
def test_deleting_the_smallest_element_produces_a_larger_intset(imp):
    assume(imp.size() > 1)
    i = imp[0]
    imper = imp.discard(i)
    assert_order(imp, imper)


@example(IntSet.interval(0, 3), 1)
@example(IntSet.interval(0, 4), 2)
@given(IntSets, integers_in_range)
def test_deleting_an_internal_element_produces_a_larger_intset(imp, i):
    assume(0 < i + 1 < imp.size())
    imper = imp.discard(imp[i])
    assert_order(imp, imper)


@given(IntSets, integers_in_range)
def test_never_equal_to_inserting_a_new_element(imp, i):
    assume(i not in imp)
    assert imp != imp.insert(i)


@given(SmallIntSets)
def test_an_intset_contains_all_its_values(imp):
    for i in imp:
        assert i in imp


@given(SmallIntSets)
def test_an_intset_iterates_in_sorted_order(imp):
    last = None
    for i in imp:
        if last is not None:
            assert i > last
        last = i


@given(st.lists(integers_in_range, min_size=2), st.randoms())
def test_always_collapses_to_an_interval(values, rnd):
    assume(min(values) < max(values))
    values.sort()
    bits = [IntSet.interval(i, j) for i, j in zip(values, values[1:])]
    rnd.shuffle(bits)
    result = IntSet.empty()
    for b in bits:
        result |= b
    assert isinstance(result, Interval)


@given(SmallIntSets)
def test_is_equal_to_sequential_insertion(imp):
    equiv = IntSet.empty()
    for i in imp:
        equiv = equiv.insert(i)
    assert imp == equiv


@given(SmallIntSets)
def test_is_equal_to_reverse_insertion(imp):
    equiv = IntSet.empty()
    for i in reversed(list(imp)):
        equiv = equiv.insert(i)
    assert imp == equiv


@given(SmallIntSets, st.randoms())
def test_is_equal_to_random_insertion(imp, rnd):
    items = list(imp)
    rnd.shuffle(items)
    equiv = IntSet.empty()
    for i in items:
        equiv = equiv.insert(i)
    assert imp == equiv


@given(SmallIntSets)
def test_an_intset_is_consistent_with_its_index(imp):
    for index, value in enumerate(imp):
        assert imp[index] == value


@given(SmallIntSets)
def test_an_intset_is_consistent_with_its_negative_index(imp):
    values = list(imp)
    for index in range(-1, -len(values) - 1, -1):
        assert values[index] == imp[index]


@given(intervals, st.lists(integers_in_range, min_size=1))
def test_insert_into_interval(bounds, ints):
    imp = IntSet.interval(*bounds)
    for i in ints:
        imp = imp.insert(i)
        assert i in imp
    for i in ints:
        assert i in imp


@given(intervals, intervals)
def test_union_of_two_intervals_contains_each_start(i1, i2):
    assume(i1[0] < i1[1])
    assume(i2[0] < i2[1])
    x = IntSet.interval(*i1) | IntSet.interval(*i2)
    assert i1[0] in x
    assert i2[0] in x


@given(interval_list, integers_in_range)
def test_unioning_a_value_in_includes_it(intervals, i):
    mp = IntSet.from_intervals(intervals)
    assume(i not in mp)
    mp2 = mp | IntSet.interval(i, i + 1)
    assert i in mp2


@example(imp=IntSet.from_iterable([0, 2]))
@given(IntSets)
def test_restricting_bounds_reduces_size_by_one(imp):
    assume(imp.size() > 0)
    lower = imp[0]
    upper = imp[-1] + 1
    pop_left = imp.restrict(lower + 1, upper)
    pop_right = imp.restrict(lower, upper - 1)
    assert pop_left.size() == imp.size() - 1
    assert pop_right.size() == imp.size() - 1


@given(SmallIntSets)
def test_restricting_bounds_splits_set(imp):
    assume(imp.size() > 0)
    lower = imp[0]
    upper = imp[-1] + 1
    for i in imp:
        left = imp.restrict(lower, i)
        right = imp.restrict(i, upper)
        assert left.size() + right.size() == imp.size()
        assert i in right
        assert i not in left
        together = left | right
        assert together.size() == imp.size()
        assert i in together


@given(IntSets, short_intervals)
def test_restricting_bounds_restricts_bounds(imp, interval):
    smaller = imp.restrict(*interval)
    assert smaller.size() <= interval[1] - interval[0]
    for i in smaller:
        assert i in imp
        assert interval[0] <= i < interval[1]


@example(IntSet.empty(), [0, 0])
@given(SmallIntSets, intervals)
def test_restricting_bounds_does_not_remove_other_items(imp, interval):
    smaller = imp.restrict(*interval)
    assert smaller.size() <= interval[1] - interval[0]
    for i in smaller:
        assert i in imp
        assert interval[0] <= i < interval[1]


@given(SmallIntSets)
def test_equality_is_preserved(imp):
    for i in imp:
        assert imp == imp.insert(i)
    assert imp == (imp | imp)


@given(st.lists(SmallIntSets, average_size=10))
def test_sorts_as_lists(intsets):
    as_lists = sorted(map(list, intsets))
    intsets.sort()
    assert as_lists == list(map(list, intsets))


@given(st.lists(SmallIntSets, average_size=10))
@example([IntSet.single(0), IntSet.single(0)])
@example([IntSet.single(0), IntSet.empty()])
@example([IntSet.interval(0, 2)])
@example([IntSet.from_iterable([1, 3, 5])])
def test_hashes_correctly(intsets):
    as_set = set(intsets)
    for i in intsets:
        assert i in as_set


@given(SmallIntSets)
def test_all_values_lie_between_bounds(imp):
    assume(imp.size() > 0)
    for i in imp:
        assert imp.start <= i < imp.end


@example(x=IntSet.empty(), y=IntSet.single(0))
@example(x=IntSet.single(0), y=IntSet.single(2))
@given(SmallIntSets, SmallIntSets)
def test_union_gives_union(x, y):
    z = x | y
    for u in (x, y):
        for t in u:
            assert t in z
    for t in z:
        assert (t in x) or (t in u)


@example(IntSet.from_iterable([1, 2, 3]), IntSet.from_iterable([2, 4, 5]))
@example(IntSet.from_iterable([0]), IntSet.from_iterable([0]))
@example(IntSet.from_iterable([0]), IntSet.from_iterable([0, 1]))
@example(IntSet.from_iterable([1, 0]), IntSet.from_iterable([0]))
@example(IntSet.from_iterable([1, 0]), IntSet.from_iterable([4, 5]))
@example(x=IntSet.from_iterable([0, 1]), y=IntSet.from_iterable([4, 5]))
@given(SmallIntSets, SmallIntSets)
def test_intersection_gives_intersection(x, y):
    assert set(x) & set(y) == set(x & y)


@example(IntSet.empty(), IntSet.empty())
@example(IntSet.single(0), IntSet.single(0))
@example(IntSet.single(0), IntSet.single(1))
@example(IntSet.interval(0, 2), IntSet.single(1))
@example(IntSet.single(1), IntSet.from_iterable([0, 2]))
@example(IntSet.single(0), IntSet.from_iterable([0, 2, 3, 4]))
@given(IntSets, IntSets)
def test_disjoint_agrees_with_intersection(x, y):
    assert x.isdisjoint(y) == (not (x & y))


@given(IntSets, integers_in_range)
def test_deleting_an_internal_element_produces_a_subset(imp, i):
    assume(0 < i + 1 < imp.size())
    impy = imp.discard(imp[i])
    assert impy.issubset(imp)


@given(IntSets)
def test_deleting_middle_element_produces_a_subset(imp):
    assume(imp.size() >= 3)
    i = imp.size() // 2
    impy = imp.discard(imp[i])
    assert impy.issubset(imp)


@given(SmallIntSets, st.randoms())
def test_deleting_an_element_proceeds_through_subsets(imp, rnd):
    elts = list(imp)
    rnd.shuffle(elts)
    current = imp
    for i in elts:
        nxt = current.discard(i)
        assert nxt.issubset(current)
        assert nxt.issubset(imp)
        current = nxt


@given(IntSets, integers_in_range)
def test_adding_an_element_produces_a_superset(imp, i):
    imper = imp.insert(i)
    assert imp.issubset(imper)


@example(IntSet.empty(), IntSet.empty())
@example(IntSet.single(0), IntSet.from_iterable([0, 2]))
@given(IntSets, IntSets)
def test_subtracting_a_superset_is_empty(x, y):
    assume(x.issubset(y))
    assert (x - y).size() == 0


@example(
    IntSet.interval(4611686018427387904, 4611686018427387907),
    IntSet.from_intervals([
        (0, 4611686018427387904), (4611686018427387904, 4611686018427387905),
        (4611686018427387906, 4611686018427387907)]))
@example(IntSet.single(0), IntSet.empty())
@example(IntSet.interval(0, 2), IntSet.from_iterable([0, 2]))
@example(IntSet.interval(0, 3), IntSet.from_iterable([0, 2]))
@example(IntSet.single(0), IntSet.from_iterable([1, 3]))
@example(IntSet.interval(0, 3), IntSet.from_iterable([0, 2, 3, 4]))
@example(IntSet.interval(0, 5), IntSet.from_iterable([0, 2]))
@example(IntSet.single(1), IntSet.from_iterable([0, 2]))
@example(IntSet.single(3), IntSet.from_iterable([0, 2]))
@given(IntSets, IntSets)
def test_subtracting_a_non_superset_is_non_empty(x, y):
    assume(not x.issubset(y))
    assert (x - y).size() > 0


@example(IntSet.empty(), IntSet.empty())
@example(IntSet.empty(), IntSet.single(0))
@example(IntSet.single(0), IntSet.single(0))
@example(IntSet.single(0), IntSet.from_iterable([0, 2]))
@example(IntSet.single(1), IntSet.from_iterable([0, 2]))
@example(IntSet.interval(0, 2), IntSet.from_iterable([0, 2]))
@example(IntSet.interval(0, 2), IntSet.from_iterable([0, 1, 3]))
@example(IntSet.interval(0, 5), IntSet.from_iterable([0, 2]))
@example(IntSet.interval(2, 4), IntSet.from_iterable([0, 2]))
@example(IntSet.interval(0, 2), IntSet.from_iterable([2, 3, 5]))
@example(IntSet.interval(0, 3), IntSet.from_iterable([2, 3, 5]))
@example(IntSet.interval(4, 7), IntSet.from_iterable([0, 1, 8]))
@example(IntSet.interval(0, 5), IntSet.from_iterable([0, 2]))
@example(
    IntSet.from_iterable([2 ** 63 + 1, 2 ** 63 + 3]),
    IntSet.from_iterable([2 ** 62 + 1, 2 ** 62 + 3]))
@given(SmallIntSets, SmallIntSets)
def test_subtraction_gives_subtraction(x, y):
    assert set(x) - set(y) == set(x - y)


@given(SmallIntSets, SmallIntSets)
def test_subtraction_cancels_union(x, y):
    assert (x - y) == (x | y) - y
    assert (x - y) | y == x | y


@example(x=IntSet.empty(), y=IntSet.empty(), z=IntSet.empty())
@given(SmallIntSets, SmallIntSets, SmallIntSets)
def test_intersection_distributes_over_union(x, y, z):
    assert x & (y | z) == (x & y) | (x & z)


@pytest.mark.parametrize('f', [op.and_, op.or_, op.xor])
@given(SmallIntSets, SmallIntSets, SmallIntSets)
def test_associative_operators(f, x, y, z):
    assert f(f(x, y), z) == f(x, f(y, z))


@pytest.mark.parametrize('f', [op.and_, op.or_, op.xor])
@example(IntSet.single(1), IntSet.single(0))
@example(IntSet.from_iterable([0, 2]), IntSet.interval(0, 2))
@example(IntSet.from_iterable([0, 2]), IntSet.from_iterable([0, 2, 3, 4]))
@example(IntSet.from_iterable([0, 34, 35]), IntSet.from_iterable([0, 32, 33]))
@example(
    IntSet.from_iterable([0, 9223372036854775811, 9223372036854775812]),
    IntSet.from_iterable([0, 9223372036854775808, 9223372036854775809]))
@example(
    IntSet.from_intervals(
        [(0, 1), (9223372036854775791, 9223372036854775792),
         (9223372036854775792, 9223372036854775808)]),
    IntSet.from_iterable([9223372036854775808, 9223372036854775810]))
@example(
    IntSet.from_intervals(
        [(0, 1), (9943224696285111261, 9943224696285111296),
         (9943224696285111296, 9943224696285111297)]),
    IntSet.from_iterable([0, 9943224696285111296, 9943224696285111297]))
@example(IntSet.interval(0, 2), IntSet.from_iterable([0, 1]))
@given(SmallIntSets, SmallIntSets)
def test_commutative_operators(f, x, y):
    assert f(x, y) == f(y, x)


@example(IntSet.single(1), IntSet.single(0))
@example(IntSet.empty(), IntSet.single(0))
@given(IntSets, SmallIntSets)
def test_subtract_is_sequential_discard(x, y):
    expected = x
    for u in y:
        expected = expected.discard(u)
    assert (x - y) == expected


@example(IntSet.interval(0, 2 ** 63 - 1))
@example(IntSet.empty())
@example(IntSet.single(1))
@given(IntSets)
def test_truthiness_is_non_emptiness(imp):
    assert bool(imp) == (imp.size() > 0)


@example(IntSet.single(1))
@example(IntSet.empty())
@example(IntSet.from_iterable([1, 3]))
@example(IntSet.interval(1, 10))
@example(IntSet.interval(1, 10) | IntSet.interval(20, 30))
@given(IntSets)
def test_repr_evals_to_self(imp):
    assert eval(repr(imp)) == imp


@example(IntSet.empty())
@example(IntSet.single(1))
@example(IntSet.interval(1, 2))
@example(IntSet.from_iterable([1, 3, 5]))
@given(SmallIntSets)
def test_reversible_as_list(imp):
    assert list(reversed(imp)) == list(reversed(list(imp)))


class SetModel(RuleBasedStateMachine):
    intsets = Bundle('IntSets')
    values = Bundle('values')

    @rule(target=values, i=integers_in_range)
    def int_value(self, i):
        return i

    @rule(target=values, i=integers_in_range, imp=intsets)
    def endpoint_value(self, i, imp):
        if len(imp[0]) > 0:
            return imp[0][-1]
        else:
            return i

    @rule(target=values, i=integers_in_range, imp=intsets)
    def startpoint_value(self, i, imp):
        if len(imp[0]) > 0:
            return imp[0][0]
        else:
            return i

    @rule(target=intsets, bounds=short_intervals)
    def build_interval(self, bounds):
        return (IntSet.interval(*bounds), list(range(*bounds)))

    @rule(target=intsets, v=values)
    def single_value(self, v):
        return (IntSet.single(v), [v])

    @rule(target=intsets, v=values)
    def adjacent_values(self, v):
        assume(v + 1 <= 2 ** 64)
        return (IntSet.interval(v, v + 2), [v, v + 1])

    @rule(target=intsets, v=values)
    def three_adjacent_values(self, v):
        assume(v + 2 <= 2 ** 64)
        return (IntSet.interval(v, v + 3), [v, v + 1, v + 2])

    @rule(target=intsets, v=values)
    def three_adjacent_values_with_hole(self, v):
        assume(v + 2 <= 2 ** 64)
        return (IntSet.single(v) | IntSet.single(v + 2), [v, v + 2])

    @rule(target=intsets, x=intsets, y=intsets)
    def union(self, x, y):
        return (x[0] | y[0], sorted(set(x[1] + y[1])))

    @rule(target=intsets, x=intsets, y=intsets)
    def intersect(self, x, y):
        return (x[0] & y[0], sorted(set(x[1]) & set(y[1])))

    @rule(target=intsets, x=intsets, y=intsets)
    def subtract(self, x, y):
        return (x[0] - y[0], sorted(set(x[1]) - set(y[1])))

    @rule(target=intsets, x=intsets, i=values)
    def insert(self, x, i):
        return (x[0].insert(i), sorted(set(x[1] + [i])))

    @rule(target=intsets, x=intsets, i=values)
    def discard(self, x, i):
        return (x[0].discard(i), sorted(set(x[1]) - set([i])))

    @rule(target=intsets, source=intsets, bounds=intervals)
    def restrict(self, source, bounds):
        return (
            source[0].restrict(*bounds), [x for x in source[1]
                                          if bounds[0] <= x < bounds[1]])

    @rule(target=intsets, x=intsets)
    def peel_left(self, x):
        if len(x[0]) == 0:
            return x
        return self.restrict(x, (x[0].start + 1, x[0].end))

    @rule(target=intsets, x=intsets)
    def peel_right(self, x):
        if len(x[0]) == 0:
            return x
        return self.restrict(x, (x[0].start, x[0].end - 1))

    @rule(x=intsets, y=intsets)
    def validate_order(self, x, y):
        assert (x[0] <= y[0]) == (x[1] <= y[1])

    @rule(x=intsets, y=intsets)
    def validate_equality(self, x, y):
        assert (x[0] == y[0]) == (x[1] == y[1])

    @rule(source=intsets)
    def validate(self, source):
        assert list(source[0]) == source[1]
        assert len(source[0]) == len(source[1])
        for i in range(-len(source[0]), len(source[0])):
            assert source[0][i] == source[1][i]
        if len(source[0]) > 0:
            for v in source[1]:
                assert source[0].start <= v < source[0].end


TestState = SetModel.TestCase


def test_validates_bounds():
    with pytest.raises(ValueError):
        IntSet.single(-1)
    with pytest.raises(ValueError):
        IntSet.single(2 ** 64)
    with pytest.raises(ValueError):
        IntSet.interval(1, 2 ** 65)
    with pytest.raises(ValueError):
        IntSet.interval(2 ** 65, 1)


def test_validates_argument_types():
    with pytest.raises(TypeError):
        IntSet.single('foo')


def test_can_produce_whole_range_intset():
    assert IntSet.interval(0, 2 ** 64).size() == 2 ** 64
