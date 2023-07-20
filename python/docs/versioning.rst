.. _versioning:

Versioning
==========

The version of the ml-wrappers package is defined in the ``version.py`` file located in the ``python/ml_wrappers`` directory. The version is specified using three variables: ``_major``, ``_minor``, and ``_patch``. 

.. code-block:: python

    name = 'ml_wrappers'
    _major = '0'
    _minor = '4'
    _patch = '12'
    version = '{}.{}.{}'.format(_major, _minor, _patch)

The version follows the format of ``major.minor.patch``. 

- ``major``: This is incremented for major changes or redesigns in the package.
- ``minor``: This is incremented for minor changes or additions of new features.
- ``patch``: This is incremented for bug fixes or minor improvements.

When ready to release a new version, create a separate PR in ml-wrappers to bump up the version in the ``version.py`` file. In the notes, make sure to mention all of the changes that have been introduced since the last release. 

.. code-block:: python

    _major = '0'
    _minor = <enter new minor version here>
    _patch = <enter new patch version here>

After the PR has been merged, checkout the master branch and get the latest code. For more details on the release process, refer to the `Release Process <release-process.html>`_ section.