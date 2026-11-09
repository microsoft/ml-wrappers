.. _versioning:

Versioning and Documentation Versions
=====================================

The package version is defined in ``python/ml_wrappers/version.py``:

.. literalinclude:: ../ml_wrappers/version.py
   :language: python

The documentation title is derived from installed package metadata during
the build. To check your installed version:

.. code-block:: bash

    python -c "from importlib.metadata import version; print(version('ml-wrappers'))"

Choose the right documentation
------------------------------

``latest`` describes the ``main`` branch and may include changes not yet
released on PyPI. ``stable`` normally tracks the newest eligible release tag;
named versions build their own tags when activated in Read the Docs.
A successful build of one version does not prove the others are current.

Older tags, including ``v0.6.4``, retain a retired Read the Docs operating
system setting. Their builds can fail even while ``latest`` succeeds.
``stable`` may therefore serve an older successful build and an unbuilt
version URL may return 404. The next approved release containing the fixed
configuration can restore release documentation. Existing tags should not
be rewritten as part of a documentation fix.

Python support and releases
---------------------------

The ``main`` branch is tested on Python 3.10 through 3.14. Optional framework
coverage varies with Python and platform; see :ref:`dependencies`.
Published PyPI metadata describes that published artifact and may lag
unreleased repository changes. A badge is not a guarantee of every optional
framework's compatibility.

Maintainers should use the
`release process <https://github.com/microsoft/ml-wrappers/blob/main/docs/release-process.md>`_
for a version-bump PR, Test/Prod workflow runs, tags and documentation checks.
