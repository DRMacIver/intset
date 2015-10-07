IntSet
======

IntSets are an efficient immutable representation of sets of unsigned 64-bit
integers with fast boolean operations and fast indexing of the set in sorted
order. They are designed to be particularly efficient for representing sets
with large contiguous ranges, so for example representing the set of all 64-bit
integers takes only a handful of bytes.

Their behaviour and API are somehwere in between that of frozenset and that of
a sorted list of deduplicated integers.
