.. _dependencies:

Dependencies
============

The ml-wrappers library has several dependencies that are required for it to function correctly. These dependencies are listed in various files throughout the repository. Here are the main dependencies:

python/ml_wrappers.egg-info/dependency_links.txt
------------------------------------------------

This file does not list any specific dependencies.

requirements-test.txt
---------------------

- pytest
- pytest-cov
- rai-test-utils==0.3.0

requirements-linting.txt
------------------------

- flake8==4.0.1
- flake8-bugbear==21.11.29
- flake8-blind-except==0.1.1
- flake8-breakpoint
- flake8-builtins==1.5.3
- flake8-logging-format==0.6.0
- flake8-pytest-style
- isort

python/ml_wrappers.egg-info/requires.txt
----------------------------------------

- numpy
- pandas
- scipy
- scikit-learn

requirements-dev.txt
--------------------

- lightgbm
- xgboost
- catboost
- tensorflow
- shap
- transformers<4.40.0
- datasets
- raiutils
- fastai
- vision_explanation_methods
- mlflow
- joblib<1.3.0; python_version <= '3.7'
- scikeras
- openai; python_version >= '3.7'

requirements-automl.txt
-----------------------

- mlflow
- azureml-automl-dnn-vision
- vision_explanation_methods

Please note that the versions of these dependencies are subject to change and it is always a good idea to check the latest version of the library for the most up-to-date information.