IntSet
======

IntSets are an efficient immutable representation of sets of unsigned 64-bit
integers with fast boolean operations and fast indexing of the set in sorted
order. They are designed to be particularly efficient for representing sets
with large contiguous ranges, so for example representing the set of all 64-bit
integers takes only a handful of bytes.

Their behaviour and API are somewhere in between that of frozenset and that of
a sorted list of deduplicated integers.

The implementation is heavily based on `Fast Mergeable Integer Maps <http://ittc.ku.edu/~andygill/papers/IntMap98.pdf>`_
by Okasaki and Gill, but it has been adapted to support a somewhat different feature
set and a more compact representation for certain usage patterns.

For usage, see the `API documentation <https://intset.readthedocs.io/en/latest/>`_.
