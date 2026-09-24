IntSet
======

IntSets are an efficient immutable representation of sets of unsigned 64-bit
integers with fast boolean operations and fast indexing of the set in sorted
order. They are designed to be particularly efficient for representing sets
with large contiguous ranges, so for example representing the set of all 64-bit
integers takes only a handful of bytes.

Their behaviour and API are somewhere in between that of frozenset and that of
a sorted list of deduplicated integers.

The implementation is heavily based on `Fast Mergeable Integer Maps
<https://web.archive.org/web/20210615200151/https://ittc.ku.edu/~andygill/papers/IntMap98.pdf>`_
by Okasaki and Gill, but it has been adapted to support a somewhat different
feature set and a more compact representation for certain usage patterns.

Installation
------------

.. code-block:: console

    pip install intset

intset supports Python 3.10 and later, and has no dependencies.

Usage
-----

.. code-block:: python

    >>> from intset import IntSet
    >>> x = IntSet([1, 2, 3, (16, 32)])
    >>> x
    IntSet([(1, 4), (16, 32)])
    >>> 20 in x
    True
    >>> len(x)
    19
    >>> x[-1]
    31
    >>> x | IntSet.interval(4, 16)
    IntSet([(1, 32)])
    >>> (~x).size() == 2 ** 64 - 19
    True

For more, see the `API documentation <https://intset.readthedocs.io/en/latest/>`_.

Development
-----------

Development uses `uv <https://docs.astral.sh/uv/>`_ and `just
<https://just.systems/>`_. Run ``just`` to list the available recipes, and
``just check`` to run everything that continuous integration runs.
