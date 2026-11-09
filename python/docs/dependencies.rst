.. _dependencies:

Dependencies
============

Install the base package with ``python -m pip install ml-wrappers``.
Its runtime dependencies are NumPy, packaging, pandas, SciPy and scikit-learn,
defined in
`python/setup.py <https://github.com/microsoft/ml-wrappers/blob/main/python/setup.py>`_.
Generated ``.egg-info`` files are build artifacts, not maintained requirements.

Optional frameworks are not installed with the base package. Install the
frameworks required by your scenario; see :ref:`supported_frameworks`.
There are no framework-specific installation extras.
For tabular ONNX inference, install ``onnxruntime>=1.19``. The ``onnx`` and
``skl2onnx`` packages are only needed to construct or convert models, not
to wrap an existing runtime session.

Repository environments
------------------------

Use requirements files from the same checkout as the code. Their version and
Python markers are authoritative; do not copy old pinned lists from a guide.

* `requirements-dev.txt <https://github.com/microsoft/ml-wrappers/blob/main/requirements-dev.txt>`_
  contains the broad framework/test development stack, including Python-specific
  NumPy, Transformers and MLflow constraints.
* `requirements-test.txt <https://github.com/microsoft/ml-wrappers/blob/main/requirements-test.txt>`_
  contains test runner and coverage dependencies.
* `requirements-linting.txt <https://github.com/microsoft/ml-wrappers/blob/main/requirements-linting.txt>`_
  contains flake8, its plugins and isort.
* `requirements-automl.txt <https://github.com/microsoft/ml-wrappers/blob/main/requirements-automl.txt>`_
  contains TensorFlow, AzureML AutoML vision and vision explanation dependencies.
* `requirements-doc.txt <https://github.com/microsoft/ml-wrappers/blob/main/requirements-doc.txt>`_
  contains Sphinx, the Read the Docs theme and documentation dependencies.

For framework tests, follow the corresponding GitHub Actions workflow,
including its PyTorch/torchvision installation and platform constraints.
Do not install every requirements file just to build documentation.
TensorFlow and SciKeras are excluded from development requirements on Python
3.14; AutoML dependencies have their own supported environments.

For a lightweight documentation environment, follow :ref:`contributing`.
Core quickstarts need no optional framework or remote inference service.
