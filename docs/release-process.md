# Release process

Releasing requires maintainer authorization and repository publishing secrets.
A documentation PR does not authorize publishing a package or changing release
tags. Do not rewrite an existing release tag to repair documentation.

## Prepare an approved release

1. Open a version-bump PR against `main`, updating
   [version.py](../python/ml_wrappers/version.py). Review the package description,
   dependency constraints and changelog/release notes for the intended release.
2. Merge only after required tests and the Documentation workflow pass. Confirm
   the Read the Docs `latest` build uses the approved `main` commit.
3. Identify the exact release commit and record its SHA. Run Test publishing
   against the approved ref before Prod. If using a branch ref, verify it has
   not moved between runs; a tag ref must point at the same reviewed commit.

## Use the publishing workflow

The [Release ml-wrappers to PyPI workflow](../.github/workflows/release-ml-wrappers.yml)
is manually dispatched from GitHub Actions. Select the approved ref and set
`releaseType` to **Test** first, then **Prod** only after successful validation.
It builds the source distribution and wheel on Python 3.11, installs the wheel,
runs `tests/main`, uploads build artifacts and publishes to the selected index.
The workflow does not bump the version, create tags or create GitHub Releases.

Test publishing uses `TEST_PYPI_API_TOKEN_ML_WRAPPERS`; production publishing uses
`PYPI_API_TOKEN_ML_WRAPPERS`. Maintainers configure those in GitHub Actions
secrets. Never put tokens, passwords or a populated `.pypirc` in the repository.

After Test succeeds, inspect the workflow artifacts and TestPyPI project page.
Smoke-test the exact candidate version in a clean environment. For example,
replace `X.Y.Z` below with the candidate version:

```bash
python -m pip install numpy packaging pandas scipy scikit-learn
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ ml-wrappers==X.Y.Z
python -c "from importlib.metadata import version; from ml_wrappers import DatasetWrapper, wrap_model; print(version('ml-wrappers'))"
```

Installing dependencies from the normal index first avoids combining a test
index and production index as competing dependency sources. Use a disposable
environment and run the documented quickstart as well. After Prod, verify the
published version and installation from the normal PyPI index.

## Tag, release and verify documentation

Create the `vX.Y.Z` tag and GitHub Release only with maintainer approval, pointing
at the same approved commit used for publishing. Coordinate their timing with
the release; a GitHub Release alone does not publish to PyPI.

Read the Docs uses configuration from the ref being built:

- `latest` builds `main`, not necessarily the installed PyPI release.
- `stable` normally follows the newest eligible release tag.
- `vX.Y.Z` builds that tag when the version is activated in Read the Docs.

Before creating a tag, verify that its commit includes a currently supported
`.readthedocs.yaml`, documentation requirements and warning-free docs.
After creating/activating the release version, check the `latest`, `stable`
and `vX.Y.Z` build logs, commit SHAs, displayed version, API content and
HTML/PDF/EPUB downloads. A successful `latest` build does not prove that
`stable` or a tag builds successfully.

Older tags, including `v0.6.4`, retain the retired Ubuntu 20.04 build setting.
Changing `main` does not repair those immutable tags: `stable` can keep serving
an older successful build and an unbuilt version URL can return 404. The next
approved release must include the repaired configuration from `main`.
Historical-version recovery requires a separate maintainer decision; do not
silently move tags or advertise missing version URLs as working.

If GitHub events no longer trigger builds, use the integration recovery
instructions in the [contributor guide](../python/docs/contributing.rst).
