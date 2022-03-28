# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function"""

import pandas as pd
import pytest
from common_utils import (create_keras_classifier, create_keras_regressor,
                          create_lightgbm_classifier,
                          create_lightgbm_regressor,
                          create_pytorch_multiclass_classifier,
                          create_pytorch_regressor,
                          create_scikit_keras_multiclass_classifier,
                          create_scikit_keras_regressor,
                          create_sklearn_linear_regressor,
                          create_sklearn_logistic_regressor,
                          create_xgboost_classifier, create_xgboost_regressor)
from constants import DatasetConstants
from ml_wrappers import wrap_model
from ml_wrappers.dataset.dataset_wrapper import DatasetWrapper
from wrapper_validator import (validate_wrapped_classification_model,
                               validate_wrapped_regression_model)


@pytest.mark.usefixtures('clean_dir')
class TestModelWrapper(object):
    def test_wrap_sklearn_logistic_regression_model(self, iris):
        train_classification_model_numpy(
            create_sklearn_logistic_regressor, iris)
        train_classification_model_pandas(
            create_sklearn_logistic_regressor, iris)
        train_classification_model_numpy(
            create_sklearn_logistic_regressor, iris,
            use_dataset_wrapper=False)
        train_classification_model_pandas(
            create_sklearn_logistic_regressor, iris,
            use_dataset_wrapper=False)

    def test_wrap_pytorch_classification_model(self, iris):
        train_classification_model_numpy(
            create_pytorch_multiclass_classifier, iris)
        train_classification_model_numpy(
            create_pytorch_multiclass_classifier, iris,
            use_dataset_wrapper=False)

    def test_wrap_xgboost_classification_model(self, iris):
        train_classification_model_numpy(create_xgboost_classifier, iris)
        train_classification_model_pandas(create_xgboost_classifier, iris)

    def test_wrap_lightgbm_classification_model(self, iris):
        train_classification_model_numpy(create_lightgbm_classifier, iris)
        train_classification_model_pandas(create_lightgbm_classifier, iris)

    def test_wrap_keras_classification_model(self, iris):
        train_classification_model_numpy(create_keras_classifier, iris)
        train_classification_model_pandas(create_keras_classifier, iris)

    def test_wrap_scikit_keras_classification_model(self, iris):
        train_classification_model_numpy(create_scikit_keras_multiclass_classifier, iris)
        train_classification_model_pandas(create_scikit_keras_multiclass_classifier, iris)

    def test_wrap_sklearn_linear_regression_model(self, housing):
        train_regression_model_numpy(
            create_sklearn_linear_regressor, housing)
        train_regression_model_pandas(
            create_sklearn_linear_regressor, housing)
        train_regression_model_numpy(
            create_sklearn_linear_regressor, housing,
            use_dataset_wrapper=False)
        train_regression_model_pandas(
            create_sklearn_linear_regressor, housing,
            use_dataset_wrapper=False)

    def test_wrap_pytorch_regression_model(self, housing):
        train_regression_model_numpy(
            create_pytorch_regressor, housing)

    def test_wrap_xgboost_regression_model(self, housing):
        train_regression_model_numpy(create_xgboost_regressor, housing)
        train_regression_model_pandas(create_xgboost_regressor, housing)

    def test_wrap_lightgbm_regression_model(self, housing):
        train_regression_model_numpy(create_lightgbm_regressor, housing)
        train_regression_model_pandas(create_lightgbm_regressor, housing)

    def test_wrap_keras_regression_model(self, housing):
        train_regression_model_numpy(create_keras_regressor, housing)
        train_regression_model_pandas(create_keras_regressor, housing)

    def test_wrap_scikit_keras_regression_model(self, housing):
        train_regression_model_numpy(create_scikit_keras_regressor, housing)
        train_regression_model_pandas(create_scikit_keras_regressor, housing)


def train_classification_model_numpy(model_initializer, dataset,
                                     use_dataset_wrapper=True):
    X_train = dataset[DatasetConstants.X_TRAIN]
    X_test = dataset[DatasetConstants.X_TEST]
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped, model_task='classification')
    validate_wrapped_classification_model(wrapped_model, X_test)


def train_classification_model_pandas(model_initializer, dataset,
                                      use_dataset_wrapper=True):
    X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                           columns=dataset[DatasetConstants.FEATURES])
    X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                          columns=dataset[DatasetConstants.FEATURES])
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped, model_task='classification')
    validate_wrapped_classification_model(wrapped_model, X_test)


def train_regression_model_numpy(model_initializer, dataset,
                                 use_dataset_wrapper=True):
    X_train = dataset[DatasetConstants.X_TRAIN]
    X_test = dataset[DatasetConstants.X_TEST]
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped, model_task='regression')
    validate_wrapped_regression_model(wrapped_model, X_test)


def train_regression_model_pandas(model_initializer, dataset,
                                  use_dataset_wrapper=True):
    X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                           columns=dataset[DatasetConstants.FEATURES])
    X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                          columns=dataset[DatasetConstants.FEATURES])
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped, model_task='regression')
    validate_wrapped_regression_model(wrapped_model, X_test)
