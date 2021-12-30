# Release process for ml-wrappers

When ready to release, create a separate PR in ml-wrappers to bump up the version in the version.py file under the python/ml_wrappers directory:

```
_major = '0'
_minor = <enter new minor version here>
_patch = <enter new patch version here>
```

In the notes make sure to mention all of the changes that have been introduced since the last release.  Usually you can take the main description in the PR.

After the PR has been merged, checkout the master branch and get the latest code.

## Release notes

On the main page, click on releases, and select "Draft a new release".

In "tag version", enter the version in the format v0.*.*, for example v0.10.0.  Keep the target as master branch.

In release title, enter either "Patch release v0.*.*" or "Release v0.*.*".

In the release notes, enter the same release notes as in the PR above for all changes that have been made to the package.

## PyPI release

For a guide on the PyPI release process, please see:

https://packaging.python.org/tutorials/packaging-projects/

### PyPI file

Create a .pypirc file in the users home directory, it should look similar to:

```
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
repository: https://upload.pypi.org/legacy/
username: interpret-community
password: PASSWORD_REMOVED

[pypitest]
repository: https://test.pypi.org/legacy/
username: interpret-community
password: PASSWORD_REMOVED
```

Note interpret-community PyPI user is currently used to publish ml-wrappers to PyPI but this may change in the future.

### Clean repo

Make sure the repo is clean prior to release on the master branch, run:

```
git clean -fdx
```

### Creating wheel

Generate the wheel file.  First activate your release environment, this can be any conda environment on the release machine:
```
conda activate my_env
```
Then update setuptools and wheel, always make sure you have the latest version installed before releasing to PyPI:
```
pip install --upgrade setuptools wheel
```
Generate the wheel where setup.py is located:
```
cd (ml-wrappers location)\python
python setup.py sdist bdist_wheel
```
If using WSL, it may be necessary to use
```
python setup.py sdist bdist_wheel --bdist-dir ~/temp/bdistwheel
```
You should see the following files in the dist directory:
```
dist/
  ml-wrappers-0.0.1-py3-none-any.whl
  ml-wrappers-0.0.1.tar.gz
```

Upgrade twine before uploading to PyPI:
```
pip install --upgrade twine
```

Note: you may need to specify --user on some environments:
```
pip install --user --upgrade twine
```

Run twine upload to the PyPI test repository:
```
twine upload --repository pypitest dist/*
```
The twine install location may not be on PATH by default; either add it or call twine using its full path.

Validate that the page looks correct on the PyPI release page.

OPTIONAL:
You can install and validate the package locally:

pip install --index-url https://test.pypi.org/simple/ --no-deps ml-wrappers

Run twine upload to the PyPI repository:
```
twine upload --repository pypi dist/*
```