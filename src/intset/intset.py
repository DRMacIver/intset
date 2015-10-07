# coding=utf-8

# This file is part of intset (https://github.com/DRMacIver/intset)

# Copyright (C) 2013-2015 David R. MacIver (david@drmaciver.com)

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.


__all__ = [
    'IntSet',
]

from abc import abstractmethod
from collections import Sequence, Set


class IntSet(object):
    """
    An IntSet is a compressed representation of a sorted list of unsigned
    64-bit integers with fast membership, union and range restriction.
    """
    @classmethod
    def empty(cls):
        """Return an empty IntSet."""
        return empty_intset

    @classmethod
    def single(cls, value):
        """Return an IntSet containing only the single value provided."""
        _validate_integer_in_range('value', value)
        _validate_integer_in_range('value + 1', value + 1)
        return _single(value)

    @classmethod
    def interval(cls, start, end):
        """
        Return an IntSet containing only the values x such that
        start <= x < end
        """
        _validate_integer_in_range('start', start)
        if end != 0:
            _validate_integer_in_range('end - 1', end - 1)
        return _interval(start, end)

    @classmethod
    def from_iterable(self, values):
        result = empty_intset
        for i in values:
            result = result._insert(i)
        return result

    @classmethod
    def from_intervals(cls, intervals):
        """Return a new IntSet which contains precisely the intervals passed
        in."""
        base = empty_intset
        for ints in intervals:
            base |= cls.interval(*ints)
        return base

    def size(self):
        """This returns the same as len() when the latter is defined, but
        IntSet may have more values than will fit in the size of index that len
        will allow."""
        return self._size

    @abstractmethod
    def restrict(self, start, end):
        """Return a new IntSet with all values x in self such that start <=
        x < end."""

    def insert(self, value):
        """Returns an IntSet which contains all the values of the current one
        plus the provided value."""
        _validate_integer_in_range('value', value)
        return self._insert(value)

    def discard(self, value):
        """Returns an IntSet which contains all the values of the current one
        except for the passed in value.

        Returns self if the value is not present rather than raising an
        error
        """
        _validate_integer_in_range('value', value)
        return self._discard(value)

    @abstractmethod
    def _insert(self, value):
        """Implementation of insert"""

    @abstractmethod
    def _discard(self, value):
        """Implementation of discard"""

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __nonzero__(self):
        return self.__bool__()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, IntSet):
            return False
        if self._size != other._size:
            return False
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __cmp__(self, other):
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
        if isinstance(self, Empty):
            return False
        while not isinstance(self, Interval):
            assert isinstance(self, Split)
            if _is_zero(i, self.mask):
                self = self.left
            else:
                self = self.right
        assert isinstance(self, Interval)
        return self.start <= i < self.end

    def __iter__(self):
        for start, end in self.intervals():
            for i in range(start, end):
                yield i

    def __getitem__(self, i):
        if i < -self._size or i >= self._size:
            raise IndexError('IntSet index %d out of range for size %d' % (
                i, self._size,
            ))
        if i < 0:
            i += self._size
        assert i >= 0
        while isinstance(self, Split):
            if i < self.left._size:
                self = self.left
            else:
                i -= self.left._size
                self = self.right
        assert isinstance(self, Interval)
        assert 0 <= i < self._size
        return self.start + i

    def __hash__(self):
        return hash((
            self._size, self.start, self.end
        ))

    def isdisjoint(self, other):
        """Returns True if self and other have no common elements."""
        if not (self and other):
            return True
        if self.start >= other.end:
            return True
        if other.start >= self.end:
            return True
        if isinstance(self, Interval) and isinstance(other, Interval):
            return False
        if isinstance(self, Interval):
            other, self = self, other
        assert isinstance(self, Split)
        return self.left.isdisjoint(other) and self.right.isdisjoint(other)

    def issubset(self, other):
        """Returns True if every element of self is also in other."""
        if not self:
            return True
        if not other:
            return False
        if isinstance(other, Interval):
            return other.start <= self.start < self.end <= other.end
        if self.start >= other.end:
            return False
        if other.start >= self.end:
            return False
        if isinstance(self, Interval):
            if self._size == 1:
                return self.start in other
            elif self._size == 2:
                return self.start in other and self.end - 1 in other
            self = self._split_interval()
        assert isinstance(self, Split)
        assert isinstance(other, Split)
        if _shorter(self.mask, other.mask):
            return False
        elif _shorter(other.mask, self.mask):
            if _is_zero(self.prefix, other.mask):
                return self.issubset(other.left)
            else:
                return self.issubset(other.right)
        else:
            # If they have incompatible prefixes the above start/end checks
            # must have returned False already because they're actually
            # disjoint.
            assert self.prefix == other.prefix
            return (
                self.left.issubset(other.left) and
                self.right.issubset(other.right))

    def __and__(self, other):
        if min(self._size, other._size) == 0:
            return empty_intset
        if self._size > other._size:
            self, other = other, self
        if self._size == 1:
            if self.start in other:
                return self
            else:
                return empty_intset
        if isinstance(self, Interval):
            return other.restrict(self.start, self.end)
        if isinstance(other, Interval):
            return self.restrict(other.start, other.end)
        assert isinstance(self, Split)
        assert isinstance(other, Split)
        if self.start > other.end:
            return empty_intset
        if self.end < other.start:
            return empty_intset
        if _shorter(other.mask, self.mask):
            self, other = other, self
        if _shorter(self.mask, other.mask):
            if _no_match(other.prefix, self.prefix, self.mask):
                return empty_intset
            elif _is_zero(other.prefix, self.mask):
                return self.left & other
            else:
                return self.right & other
        else:
            assert self.mask == other.mask
            if self.prefix == other.prefix:
                return self._new_split(
                    self.prefix, self.mask,
                    self.left & other.left,
                    self.right & other.right
                )
            else:
                return empty_intset

    def __sub__(self, other):
        if other._size == 0:
            return self
        if self._size == 0:
            return self
        if isinstance(other, Interval):
            return self.restrict(self.start, other.start) | \
                self.restrict(other.end, self.end)
        if self._size == 1:
            if self.start in other:
                return empty_intset
            else:
                return self
        if isinstance(self, Interval):
            self = self._split_interval()
        assert isinstance(self, Split)
        assert isinstance(other, Split)
        if _shorter(self.mask, other.mask):
            if _no_match(other.prefix, self.prefix, self.mask):
                return self
            elif _is_zero(other.prefix, self.mask):
                return self._new_split(
                    self.prefix, self.mask, self.left - other, self.right
                )
            else:
                return self._new_split(
                    self.prefix, self.mask, self.left, self.right - other
                )
        elif _shorter(other.mask, self.mask):
            if _is_zero(self.prefix, other.mask):
                return self - other.left
            else:
                return self - other.right
        else:
            if self.prefix == other.prefix:
                return self._new_split(
                    self.prefix, self.mask,
                    self.left - other.left,
                    self.right - other.right
                )
            else:
                return self

    def __xor__(self, other):
        return (self | other) - (self & other)

    def __or__(self, other):
        if self._size == 0:
            return other
        if other._size == 0:
            return self
        if other._size > self._size:
            other, self = self, other
        if isinstance(self, Interval) and isinstance(other, Interval):
            if not (self.start > other.end or other.start > self.end):
                return self._new_interval(
                    min(self.start, other.start), max(self.end, other.end))
            elif self._size > 1:
                return self._split_interval() | other
            else:
                assert self._size == other._size == 1
                return _join(self.start, self, other.start, other)
        if isinstance(other, Interval):
            if other.start <= self.start < self.end <= other.end:
                return other
        if isinstance(self, Interval):
            if self.start <= other.start < other.end <= self.end:
                return self
        if isinstance(other, Interval):
            if other._size == 1:
                return self._insert(other.start)
            else:
                other = other._split_interval()
        if isinstance(self, Interval):
            self = self._split_interval()
        assert isinstance(self, Split)
        assert isinstance(other, Split)
        if _shorter(other.mask, self.mask):
            self, other = other, self
        if _shorter(self.mask, other.mask):
            if _no_match(other.prefix, self.prefix, self.mask):
                return _join(
                    self.prefix, self, other.prefix, other
                )
            elif _is_zero(other.prefix, self.mask):
                return self._new_split(
                    self.prefix, self.mask, self.left | other, self.right
                )
            else:
                return self._new_split(
                    self.prefix, self.mask, self.left, self.right | other
                )
        else:
            assert self.mask == other.mask
            if self.prefix == other.prefix:
                return self._new_split(
                    self.prefix, self.mask,
                    self.left | other.left,
                    self.right | other.right
                )
            else:
                return _join(self.prefix, self, other.prefix, other)

    def intervals(self):
        """
        Provide a sorted iterator over a sequence of values start < end which
        represent non-overlapping intervals such that for any start <= x < end
        x in self
        """
        stack = [self]
        while stack:
            head = stack.pop()
            if isinstance(head, Interval):
                yield (head.start, head.end)
            elif isinstance(head, Split):
                stack.append(head.right)
                stack.append(head.left)

    def reversed_intervals(self):
        """Iterator over the reverse of intervals()"""
        stack = [self]
        while stack:
            head = stack.pop()
            if isinstance(head, Interval):
                yield (head.start, head.end)
            elif isinstance(head, Split):
                stack.append(head.left)
                stack.append(head.right)

    def __reversed__(self):
        for start, end in self.reversed_intervals():
            for i in range(end - 1, start - 1, -1):
                yield i

    def _new_split(self, prefix, mask, left, right):
        if left._size == 0:
            return right
        if right._size == 0:
            return left
        if (
            left.__class__ is right.__class__ is Interval and
            left.start <= right.end and right.start <= left.end
        ):
            return self._new_interval(left.start, right.end)
        if (
            self.__class__ is Split and
            left is self.left and right is self.right and
            prefix == self.prefix and mask == self.mask
        ):
            return self._compress()
        return Split(
            prefix=prefix, mask=mask, left=left, right=right)._compress()

    def _new_interval(self, start, end):
        return _interval(start, end)

Sequence.register(IntSet)
Set.register(IntSet)


class Empty(IntSet):
    _size = 0

    def __init__(self):
        pass

    def __hash__(self):
        return 0

    def _insert(self, value):
        return _single(value)

    def _discard(self, value):
        return self

    def restrict(self, start, end):
        return self

    def __repr__(self):
        return 'IntSet.empty()'


empty_intset = Empty()


class Split(IntSet):

    def __init__(self, prefix, mask, left, right):
        self.mask = mask
        self.prefix = prefix
        self.left = left
        self.right = right
        self._size = left._size + right._size
        self.start = left.start
        self.end = right.end

    def _compress(self):
        if self.end == self.start + self._size:
            return _interval(self.start, self.end)
        else:
            return self

    def __repr__(self):
        intervals = list(self.intervals())
        if any(i + 1 < j for i, j in intervals):
            return 'IntSet.from_intervals([%s])' % (', '.join(
                '(%d, %d)' % interval for interval in self.intervals()
            ))
        else:
            return 'IntSet.from_iterable(%r)' % ([i for i, _ in intervals],)

    def _insert(self, value):
        if _no_match(value, self.prefix, self.mask):
            return _join(
                value, _single(value),
                self.prefix, self
            )
        elif _is_zero(value, self.mask):
            return self._new_split(
                prefix=self.prefix, mask=self.mask,
                left=self.left._insert(value),
                right=self.right,
            )
        else:
            return self._new_split(
                prefix=self.prefix, mask=self.mask,
                left=self.left,
                right=self.right._insert(value),
            )

    def _discard(self, value):
        if _is_zero(value, self.mask):
            return self._new_split(
                prefix=self.prefix, mask=self.mask,
                left=self.left.discard(value), right=self.right
            )
        else:
            return self._new_split(
                prefix=self.prefix, mask=self.mask,
                left=self.left, right=self.right.discard(value)
            )

    def restrict(self, start, end):
        if (start <= self.start) and (self.end <= end):
            return self
        if start > self.end:
            return empty_intset
        if end < self.start:
            return empty_intset
        return self._new_split(
            mask=self.mask, prefix=self.prefix,
            left=self.left.restrict(start, end),
            right=self.right.restrict(start, end),
        )


class Interval(IntSet):

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self._size = end - start

    def __repr__(self):
        if self._size == 1:
            return 'IntSet.single(%d)' % (self.start,)
        else:
            return 'IntSet.interval(%d, %d)' % (self.start, self.end)

    def _insert(self, value):
        if self.start <= value < self.end:
            return self
        elif self._size == 1:
            return _join(self.start, self, value, _single(value))
        elif value + 1 == self.start:
            return _interval(self.start - 1, self.end)
        elif value == self.end:
            return _interval(self.start, self.end + 1)
        else:
            return self._split_interval()._insert(value)

    def _discard(self, value):
        if value < self.start or value >= self.end:
            return self
        if value == self.start:
            return _interval(self.start + 1, self.end)
        if value + 1 == self.end:
            return _interval(self.start, self.end - 1)
        return self._split_interval().discard(value)

    def restrict(self, start, end):
        if start <= self.start < self.end <= end:
            return self
        else:
            start = max(start, self.start)
            end = min(end, self.end)
            return _interval(max(start, self.start), min(end, self.end))

    def _split_interval(self):
        assert self._size >= 2
        split_mask = branch_mask(self.start, self.end - 1)
        split_prefix = _mask_off(self.start, split_mask)
        split_point = split_prefix | split_mask
        assert self.start < split_point < self.end
        return Split(
            prefix=split_prefix, mask=split_mask,
            left=Interval(self.start, split_point),
            right=Interval(split_point, self.end),
        )

    def _new_interval(self, start, end):
        if self.start == start and self.end == end:
            return self
        else:
            return super(Interval, self)._new_interval(start, end)


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


def _join(p1, t1, p2, t2):
    assert t1.size
    assert t2.size
    assert p1 != p2
    m = branch_mask(p1, p2)
    p = _mask_off(p1, m)
    if not _is_zero(p1, m):
        t1, t2 = t2, t1
    return Split(prefix=p, mask=m, left=t1, right=t2)


def _no_match(i, p, m):
    return _mask_off(i, m) != p


def _shorter(m1, m2):
    return m1 > m2

_UPPER_BOUND = 2 ** 64

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


def _interval(start, end):
    if end <= start:
        return empty_intset
    else:
        return Interval(start, end)


def _single(value):
    return Interval(value, value + 1)
