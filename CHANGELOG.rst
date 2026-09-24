Changelog
=========

Unreleased
----------

* Fixed ``IntSet.single(2 ** 64 - 1)`` raising ``ValueError``. The largest
  representable value could be inserted with ``insert`` or via ``interval`` but
  was wrongly rejected by ``single``.
* Fixed the import of ``Sequence`` and ``Set`` on Python 3.10 and later, where
  they are only available from ``collections.abc`` (`#5
  <https://github.com/DRMacIver/intset/pull/5>`_, thanks to Mike Salvatore).
* ``intset.__version__`` and ``intset.__version_info__`` are now exported from
  the top-level package.
* Dropped support for Python 2 and for Python 3 versions before 3.10.
* Modernised packaging (``pyproject.toml``), testing (current Hypothesis and
  pytest) and continuous integration (GitHub Actions).

1.0.0 - 2015-10-08
------------------

* Initial release.
