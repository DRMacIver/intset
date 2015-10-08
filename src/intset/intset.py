# coding=utf-8

# This file is part of intset (https://github.com/DRMacIver/intset)

# Copyright (C) 2013-2015 David R. MacIver (david@drmaciver.com)

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

__all__ = [
    'IntSet',
]

from collections import Sequence, Set


class IntSetMeta(type):

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            return self._wrap(())
        elif len(args) == 1:
            result = IntSet.Builder()
            for i in args[0]:
                try:
                    result.insert(i)
                except TypeError:
                    result.insert_interval(*i)
            return result.build()
        else:
            raise TypeError('IntSet expected at most 1 arguments, got %d' % (
                len(args),
            ))

    def _wrap(self, value):
        return type.__call__(self, value)


class IntSet(IntSetMeta('IntSet', (object,), {})):
    """
    An IntSet is a compressed immutable representation of a sorted list of
    unsigned 64-bit integers with fast membership, union and range restriction.

    It mostly behaves as if it were a sorted list of deduplicated integer
    values. In particular, you can index it if it were, and it will sort and
    compare equal (to other IntSets) as if it were.

    Note that unlike lists, intsets may feasibly have more than sys.maxint
    elements, and calling len() on such an intset may raise an OverflowError.
    If you wish to avoid this, use .size() instead.

    Because IntSet is immutable, unlike list it may also be used as a hash key.

    It also supports set operations. In particular, all the boolean operations
    are supported:

        x & y: An IntSet containing the values that are present in both x and y

        x | y: An IntSet containing the values present in either x or y

        x - y: An IntSet containing the values present in x but not y

        x ^ y: An IntSet containing the values present in x or y but not both

        ~x: An IntSet containing all values in the range 0 <= i < 2 ** 64 that
            are not present in x (IntSet can represent this efficiently. It
            won't allocate 2 ** 64 integers worth of memory).

    IntSets may be constructed either from the dedicated class methods or by
    calling the class as you usually would for a set. So IntSet([1, 2, 3]) is
    an IntSet containing the values 1, 2 and 3.

    When calling an IntSet this way, non-integer values which are iterable
    sequences of length 2 will be interpreted as intervals start <= x < end.
    So e.g. IntSet([1, [10, 100]]) will contain the numbers 1 and 10, ..., 99.
    """

    __slots__ = ('wrapped')

    class Builder(object):
        """An IntSet.Builder is for building up an IntSet incrementally through
        a series of insertions.

        This will typically be much faster than repeatedly calling
        insert on an IntSet object. The intended usage is to repeatedly
        call insert() or insert_interval() on a builder, then call
        build() at the end. Note that you can continue to insert further
        data into a Builder afterwards if you wish, and this will not
        affect previously built IntSet instances.

        """

        def __init__(self):
            self.wrapped = ()
            self.start = 0
            self.end = 0

        def insert(self, value):
            """Add a single value to the IntSet to be built."""
            if value == self.end:
                self.end += 1
            elif value + 1 == self.start:
                self.start -= 1
            else:
                self._collapse()
                self.start = value
                self.end = value + 1

        def insert_interval(self, start, end):
            """Add all values x such that start <= x < end to the IntSet to be
            built."""
            if not (self.start >= end or start >= self.end):
                self.start = min(start, self.start)
                self.end = max(end, self.end)
            else:
                self._collapse()
                self.start = start
                self.end = end

        def build(self):
            """Produce a new IntSet with all the values previously inserted to
            this builder.

            You may call build() more than once, and any values inserted
            in between those calls will also be present, but previously
            built values will be unaffected by subsequent inserts

            """
            self._collapse()
            self.start = 0
            self.end = 0
            return IntSet._wrap(self.wrapped)

        def _collapse(self):
            if self.start < self.end:
                self.wrapped = _union(
                    self.wrapped, _new_interval(self.start, self.end)
                )

    def __getstate__(self):
        # wrap in a tuple because a falsey value will cause the corresponding
        # setstate to not be called.
        return (list(self.intervals()),)

    def __setstate__(self, state):
        self.wrapped = IntSet.from_intervals(state[0]).wrapped

    def __init__(self, wrapped):
        assert isinstance(wrapped, tuple)
        self.wrapped = wrapped

    def __repr__(self):
        bits = []
        for i, j in self.intervals():
            if i + 1 < j:
                bits.append((i, j))
            else:
                bits.append(i)
        return 'IntSet(%r)' % (bits,)

    @classmethod
    def empty(cls):
        """Return an empty IntSet."""
        return IntSet._wrap(())

    @classmethod
    def single(cls, value):
        """Return an IntSet containing only the single value provided."""
        _validate_integer_in_range('value', value)
        _validate_integer_in_range('value + 1', value + 1)
        return IntSet._wrap(_new_single(value))

    @classmethod
    def interval(cls, start, end):
        """
        Return an IntSet containing only the values x such that
        start <= x < end
        """
        _validate_integer_in_range('start', start)
        if end != 0:
            _validate_integer_in_range('end - 1', end - 1)
        return IntSet._wrap(_new_maybe_empty_interval(start, end))

    @classmethod
    def from_iterable(self, values):
        """Return an IntSet containing everything in values, which should be an
        iterable over intsets in the valid range."""
        builder = IntSet.Builder()
        for i in values:
            builder.insert(i)
        return builder.build()

    @classmethod
    def from_intervals(cls, intervals):
        """Return a new IntSet which contains precisely the intervals passed
        in."""
        builder = IntSet.Builder()
        for ints in intervals:
            builder.insert_interval(*ints)
        return builder.build()

    def size(self):
        """This returns the same as len() when the latter is defined, but
        IntSet may have more values than will fit in the size of index that len
        will allow."""
        if self.wrapped:
            return self.wrapped[_SIZE]
        else:
            return 0

    def insert(self, value):
        """Returns an IntSet which contains all the values of the current one
        plus the provided value."""
        _validate_integer_in_range('value', value)
        return IntSet._wrap(_insert(self.wrapped, value))

    def discard(self, value):
        """Returns an IntSet which contains all the values of the current one
        except for the passed in value.

        Returns self if the value is not present rather than raising an
        error

        """
        _validate_integer_in_range('value', value)
        return IntSet._wrap(_discard(self.wrapped, value))

    def restrict(self, start, end):
        """Return a new IntSet with all values x in self such that start <=
        x < end."""
        return IntSet._wrap(_restrict(self.wrapped, start, end))

    def __len__(self):
        return self.size()

    def __bool__(self):
        return bool(self.wrapped)

    def __nonzero__(self):
        return self.__bool__()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, IntSet):
            return False
        if self.size() != other.size():
            return False
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __cmp__(self, other):
        if not isinstance(other, IntSet):
            raise TypeError(
                'Unorderable types IntSet and %s' % (type(other).__name__,))
        self_intervals = list(self.intervals())
        other_intervals = list(other.intervals())
        self_intervals.reverse()
        other_intervals.reverse()
        while self_intervals and other_intervals:
            self_head = self_intervals.pop()
            other_head = other_intervals.pop()
            if self_head[0] < other_head[0]:
                return -1
            if self_head[0] > other_head[0]:
                return 1
            if self_head[1] < other_head[1]:
                other_intervals.append((self_head[1], other_head[1]))
            if self_head[1] > other_head[1]:
                self_intervals.append((other_head[1], self_head[1]))
        if self_intervals:
            return 1
        if other_intervals:
            return -1
        return 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def __contains__(self, i):
        return _contains(self.wrapped, i)

    def __iter__(self):
        for start, end in self.intervals():
            for i in range(start, end):
                yield i

    def __getitem__(self, i):
        size = self.size()
        if i < -size or i >= size:
            raise IndexError('IntSet index %d out of range for size %d' % (
                i, size,
            ))
        if i < 0:
            i += size
        assert i >= 0
        return _getitem(self.wrapped, i)

    def __hash__(self):
        return hash(self.wrapped[:3])

    def __copy__(self):
        return self

    def __deepcopy__(self, table):
        return table.setdefault(self, self)

    def isdisjoint(self, other):
        """Returns True if self and other have no common elements."""
        return _isdisjoint(self.wrapped, other.wrapped)

    def intersects(self, other):
        """Returns True if there is an element i such that i in self and i in
        other."""
        return not self.isdisjoint(other)

    def issubset(self, other):
        return _issubset(self.wrapped, other.wrapped)

    def issuperset(self, other):
        """Returns True if every element of other is also in self."""
        return other.issubset(self)

    def __and__(self, other):
        assert isinstance(other, IntSet)
        return IntSet._wrap(_intersect(self.wrapped, other.wrapped))

    def __invert__(self):
        return whole_range - self

    def __sub__(self, other):
        return IntSet._wrap(_subtract(self.wrapped, other.wrapped))

    def __xor__(self, other):
        return (self | other) - (self & other)

    def __or__(self, other):
        return IntSet._wrap(_union(self.wrapped, other.wrapped))

    def intervals(self):
        """
        Provide a sorted iterator over a sequence of values start < end which
        represent non-overlapping intervals such that for any start <= x < end
        x in self
        """
        return _intervals(self.wrapped)

    def reversed_intervals(self):
        """Iterator over the reverse of intervals()"""
        return _reversed_intervals(self.wrapped)

    def __reversed__(self):
        for start, end in self.reversed_intervals():
            for i in range(end - 1, start - 1, -1):
                yield i


Sequence.register(IntSet)
Set.register(IntSet)


def _new_maybe_empty_interval(start, end):
    if end <= start:
        return ()
    return _new_interval(start, end)


_START = 0
_END = 1
_SIZE = 2
_PREFIX = 3
_MASK = 4
_LEFT = 5
_RIGHT = 6

_INTERVAL_LENGTH = 3
_SPLIT_LENGTH = 7


def _new_interval(start, end):
    return (start, end, end - start)


def _new_single(value):
    return (value, value + 1, 1)


def _new_split_maybe_empty(prefix, mask, left, right):
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    return _new_split(prefix, mask, left, right)


def _new_split(prefix, mask, left, right):
    if left[_SIZE] + right[_SIZE] + left[_START] == right[_END]:
        return _new_interval(left[_START], right[_END])
    return (
        left[_START], right[_END],
        left[_SIZE] + right[_SIZE], prefix, mask, left, right
    )


def _split_interval(ins):
    start = ins[_START]
    end = ins[_END]
    split_mask = branch_mask(start, end - 1)
    split_prefix = _mask_off(start, split_mask)
    split_point = split_prefix | split_mask
    return (
        start, end, ins[_SIZE], split_prefix, split_mask,
        _new_interval(start, split_point), _new_interval(split_point, end)
    )


def _join(p1, t1, p2, t2):
    m = branch_mask(p1, p2)
    p = _mask_off(p1, m)
    if not _is_zero(p1, m):
        t1, t2 = t2, t1
    return _new_split(p, m, t1, t2)


def _insert(ins, value):
    l = len(ins)
    if l == 0:
        return _new_single(value)
    elif l == _INTERVAL_LENGTH:
        start = ins[_START]
        end = ins[_END]
        if start <= value < end:
            return ins
        elif value == end:
            return _new_interval(start, end + 1)
        elif value + 1 == start:
            return _new_interval(value, end)
        elif ins[_SIZE] == 1:
            return _join(start, ins, value, _new_single(value))
        else:
            ins = _split_interval(ins)
    prefix = ins[_PREFIX]
    mask = ins[_MASK]
    if _no_match(value, prefix, mask):
        return _join(
            value, _new_single(value),
            prefix, ins
        )
    elif _is_zero(value, mask):
        return _new_split(
            prefix, mask, _insert(ins[_LEFT], value), ins[_RIGHT])
    else:
        return _new_split(
            prefix, mask, ins[_LEFT], _insert(ins[_RIGHT], value))


def _getitem(self, i):
    while len(self) > _INTERVAL_LENGTH:
        if i < self[_LEFT][_SIZE]:
            self = self[_LEFT]
        else:
            i -= self[_LEFT][_SIZE]
            self = self[_RIGHT]
    return self[_START] + i


def _discard(self, value):
    l = len(self)
    if l == 0:
        return self
    elif l == _INTERVAL_LENGTH:
        if value < self[_START] or value >= self[_END]:
            return self
        if value == self[_START]:
            return _new_maybe_empty_interval(self[_START] + 1, self[_END])
        if value + 1 == self[_END]:
            return _new_maybe_empty_interval(self[_START], self[_END] - 1)
        self = _split_interval(self)
    if _is_zero(value, self[_MASK]):
        return _new_split_maybe_empty(
            self[_PREFIX], self[_MASK],
            _discard(self[_LEFT], value), self[_RIGHT]
        )
    else:
        return _new_split_maybe_empty(
            self[_PREFIX], self[_MASK],
            self[_LEFT], _discard(self[_RIGHT], value)
        )


def _union(self, other):
    if len(self) == 0:
        return other
    if len(other) == 0:
        return self
    if other[_SIZE] > self[_SIZE]:
        self, other = other, self
    if len(self) == _INTERVAL_LENGTH:
        if self[_START] <= other[_START] and other[_END] <= self[_END]:
            return self
        if len(other) == _INTERVAL_LENGTH:
            if self[_START] <= other[_END] and other[_START] <= self[_END]:
                return _new_interval(
                    min(self[_START], other[_START]),
                    max(self[_END], other[_END]),
                )
            elif self[_SIZE] > 1:
                return _union(_split_interval(self), other)
            else:
                return _join(self[_START], self, other[_START], other)
    if len(other) == _INTERVAL_LENGTH:
        if other[_SIZE] == 1:
            return _insert(self, other[_START])
        else:
            other = _split_interval(other)
    if len(self) == _INTERVAL_LENGTH:
        self = _split_interval(self)
    if _shorter(other[_MASK], self[_MASK]):
        self, other = other, self
    if _shorter(self[_MASK], other[_MASK]):
        if _no_match(other[_PREFIX], self[_PREFIX], self[_MASK]):
            return _join(
                self[_PREFIX], self, other[_PREFIX], other
            )
        elif _is_zero(other[_PREFIX], self[_MASK]):
            return _new_split(
                self[_PREFIX], self[_MASK],
                _union(self[_LEFT], other), self[_RIGHT]
            )
        else:
            return _new_split(
                self[_PREFIX], self[_MASK],
                self[_LEFT], _union(self[_RIGHT], other)
            )
    else:
        assert self[_MASK] == other[_MASK]
        if self[_PREFIX] == other[_PREFIX]:
            return _new_split(
                self[_PREFIX], self[_MASK],
                _union(self[_LEFT], other[_LEFT]),
                _union(self[_RIGHT], other[_RIGHT])
            )
        else:
            return _join(self[_PREFIX], self, other[_PREFIX], other)


def _restrict(self, start, end):
    if not self:
        return self
    if start >= self[_END] or self[_START] >= end:
        return ()
    if len(self) == _INTERVAL_LENGTH:
        return _new_interval(
            max(start, self[_START]), min(end, self[_END]))
    return _new_split_maybe_empty(
        self[_PREFIX], self[_MASK],
        _restrict(self[_LEFT], start, end),
        _restrict(self[_RIGHT], start, end),
    )


def _contains(self, value):
    if not self:
        return False
    while len(self) != _INTERVAL_LENGTH:
        if _is_zero(value, self[_MASK]):
            self = self[_LEFT]
        else:
            self = self[_RIGHT]
    return self[_START] <= value < self[_END]


def _intersect(self, other):
    if not (self and other):
        return ()
    if self[_SIZE] > other[_SIZE]:
        self, other = other, self
    if other[_SIZE] == 1:
        if _contains(self, other[_START]):
            return other
        else:
            return ()
    if len(self) == _INTERVAL_LENGTH:
        return _restrict(other, self[_START], self[_END])
    if len(other) == _INTERVAL_LENGTH:
        return _restrict(self, other[_START], other[_END])
    if self[_START] > other[_END]:
        return ()
    if self[_END] < other[_START]:
        return ()
    if _shorter(other[_MASK], self[_MASK]):
        self, other = other, self
    if _shorter(self[_MASK], other[_MASK]):
        if _no_match(other[_PREFIX], self[_PREFIX], self[_MASK]):
            return ()
        elif _is_zero(other[_PREFIX], self[_MASK]):
            return _intersect(self[_LEFT], other)
        else:
            return _intersect(self[_RIGHT], other)
    else:
        assert self[_MASK] == other[_MASK]
        if self[_PREFIX] == other[_PREFIX]:
            return _new_split_maybe_empty(
                self[_PREFIX], self[_MASK],
                _intersect(self[_LEFT], other[_LEFT]),
                _intersect(self[_RIGHT], other[_RIGHT])
            )
        else:
            return ()


def _subtract(self, other):
    if not (other and self):
        return self
    if len(other) == _INTERVAL_LENGTH:
        return _union(
            _restrict(self, self[_START], other[_START]),
            _restrict(self, other[_END], self[_END]))
    if self[_SIZE] == 1:
        if _contains(other, self[_START]):
            return ()
        else:
            return self
    if len(self) == _INTERVAL_LENGTH:
        self = _split_interval(self)
    if _shorter(self[_MASK], other[_MASK]):
        if _no_match(other[_PREFIX], self[_PREFIX], self[_MASK]):
            return self
        elif _is_zero(other[_PREFIX], self[_MASK]):
            return _new_split_maybe_empty(
                self[_PREFIX], self[_MASK],
                _subtract(self[_LEFT], other), self[_RIGHT]
            )
        else:
            return _new_split_maybe_empty(
                self[_PREFIX], self[_MASK], self[_LEFT],
                _subtract(self[_RIGHT], other)
            )
    elif _shorter(other[_MASK], self[_MASK]):
        if _is_zero(self[_PREFIX], other[_MASK]):
            return _subtract(self, other[_LEFT])
        else:
            return _subtract(self, other[_RIGHT])
    else:
        if self[_PREFIX] == other[_PREFIX]:
            return _new_split_maybe_empty(
                self[_PREFIX], self[_MASK],
                _subtract(self[_LEFT], other[_LEFT]),
                _subtract(self[_RIGHT], other[_RIGHT])
            )
        else:
            return self


def _isdisjoint(self, other):
    if not (self and other):
        return True
    if self[_START] >= other[_END]:
        return True
    if other[_START] >= self[_END]:
        return True
    if len(self) == _INTERVAL_LENGTH:
        if len(other) == _INTERVAL_LENGTH:
            return False
        other, self = self, other
    return _isdisjoint(self[_LEFT], other) and _isdisjoint(self[_RIGHT], other)


def _issubset(self, other):
    if not self:
        return True
    if not other:
        return False
    if len(other) == _INTERVAL_LENGTH:
        return (
            other[_START] <= self[_START] and
            self[_END] <= other[_END])
    if self[_START] >= other[_END]:
        return False
    if other[_START] >= self[_END]:
        return False
    if len(self) == _INTERVAL_LENGTH:
        if self[_SIZE] == 1:
            return _contains(other, self[_START])
        elif self[_SIZE] == 2:
            return (
                _contains(other, self[_START]) and
                _contains(other, self[_END] - 1))
        self = _split_interval(self)
    if _shorter(self[_MASK], other[_MASK]):
        return False
    elif _shorter(other[_MASK], self[_MASK]):
        if _is_zero(self[_PREFIX], other[_MASK]):
            return _issubset(self, other[_LEFT])
        else:
            return _issubset(self, other[_RIGHT])
    else:
        # If they have incompatible prefixes the above start/end checks
        # must have returned False already because they're actually
        # disjoint.
        assert self[_PREFIX] == other[_PREFIX]
        return (
            _issubset(self[_LEFT], other[_LEFT]) and
            _issubset(self[_RIGHT], other[_RIGHT]))


def _intervals(self):
    if not self:
        return
    stack = [self]
    while stack:
        head = stack.pop()
        if len(head) == _INTERVAL_LENGTH:
            yield (head[_START], head[_END])
        else:
            stack.append(head[_RIGHT])
            stack.append(head[_LEFT])


def _reversed_intervals(self):
    if not self:
        return
    stack = [self]
    while stack:
        head = stack.pop()
        if len(head) == _INTERVAL_LENGTH:
            yield (head[_START], head[_END])
        else:
            stack.append(head[_LEFT])
            stack.append(head[_RIGHT])


def _right_fill_bits(key):
    key |= (key >> 1)
    key |= (key >> 2)
    key |= (key >> 4)
    key |= (key >> 8)
    key |= (key >> 16)
    key |= (key >> 32)
    return key


def _highest_bit_mask(k):
    k = _right_fill_bits(k)
    k ^= (k >> 1)
    return k


def branch_mask(p1, p2):
    return _highest_bit_mask(p1 ^ p2)


def _mask_off(i, m):
    return i & (~(m - 1) ^ m)


def _is_zero(i, m):
    return (i & m) == 0


def _no_match(i, p, m):
    return _mask_off(i, m) != p


def _shorter(m1, m2):
    return m1 > m2

_UPPER_BOUND = 2 ** 64

whole_range = IntSet._wrap(_new_interval(0, _UPPER_BOUND))

INTEGER_TYPES = (type(0), type(2 ** 64))


def _validate_integer_in_range(name, i):
    if not isinstance(i, INTEGER_TYPES):
        raise TypeError(
            'Expected %s to be an integer but got %r of type %s' % (
                name, i, type(i).__name__))
    if i < 0 or i >= _UPPER_BOUND:
        raise ValueError(
            'Argument %s=%d out of required range 0 <= %s < 2 ** 64' % (
                name, i, name))
