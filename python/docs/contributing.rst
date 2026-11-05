.. _contributing:

Contributing
============

We welcome contributions and suggestions! Please see the `CONTRIBUTING.md <https://github.com/microsoft/ml-wrappers/blob/main/CONTRIBUTING.md>`_ file for more details.

Feature Request
---------------

If you have a feature request related to this project, please follow the template provided in the `feature_request.md <https://github.com/microsoft/ml-wrappers/blob/main/.github/ISSUE_TEMPLATE/feature_request.md>`_ file. This template will guide you to describe the problem, the solution you'd like, any alternative solutions you've considered, and any additional context or screenshots about the feature request.

Bug Report
----------

If you encounter a bug and want to report it, please use the template provided in the `bug_report.md <https://github.com/microsoft/ml-wrappers/blob/main/.github/ISSUE_TEMPLATE/bug_report.md>`_ file. This template will guide you to describe the bug, the steps to reproduce it, the expected behavior, any screenshots, and any additional context about the problem.

Support
-------

This project uses GitHub Issues to track bugs and feature requests. Please search the existing issues before filing new issues to avoid duplicates. For new issues, file your bug or feature request as a new Issue. Support for this project, ml-wrappers, is limited to the resources listed above. For more details, please see the `SUPPORT.md <https://github.com/microsoft/ml-wrappers/blob/main/SUPPORT.md>`_ file.

Microsoft Open Source Code of Conduct
-------------------------------------

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_. For more information, see the `CODE_OF_CONDUCT.md <https://github.com/microsoft/ml-wrappers/blob/main/CODE_OF_CONDUCT.md>`_ file.

Security
--------

Microsoft takes the security of our software products and services seriously. If you believe you have found a security vulnerability in any Microsoft-owned repository, please report it to us as described in the `SECURITY.md <https://github.com/microsoft/ml-wrappers/blob/main/SECURITY.md>`_ file.

Release Process
---------------

When ready to release, create a separate PR in ml-wrappers to bump up the version in the version.py file under the python/ml_wrappers directory. For more details, please see the `release-process.md <https://github.com/microsoft/ml-wrappers/blob/main/docs/release-process.md>`_ file.

Documentation
-------------

Build the documentation from the repository root in a Python 3.11 environment:

.. code-block:: bash

    python -m pip install -r requirements-doc.txt ./python
    python -m sphinx -b html python/docs _build/html

The Documentation GitHub Actions workflow checks the HTML build on pull requests
and pushes to main. Read the Docs publishes the site using ``.readthedocs.yaml``.

If pushes stop triggering Read the Docs builds, check the repository webhook's
recent deliveries. A response saying the webhook has no secret requires a
Read the Docs project maintainer to resync the GitHub integration under
Settings > Integrations. For manually configured webhooks, copy the new shared
secret into the matching GitHub webhook's Secret field; do not commit it.
See `Read the Docs integration setup
<https://docs.readthedocs.com/platform/stable/guides/setup/git-repo-manual.html>`_
for details. After restoring the integration, trigger a build of ``latest`` and
confirm that it uses the current main commit.

License
-------

This project is licensed under the MIT License. For more details, please see the `LICENSE.txt <https://github.com/microsoft/ml-wrappers/blob/main/LICENSE.txt>`_ file.