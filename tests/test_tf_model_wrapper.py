# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for WrappedTensorflowModel"""

import pytest
from common_utils import (create_keras_classifier, create_keras_regressor,
                          create_scikit_keras_regressor)
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.model import WrappedTensorflowModel
from train_wrapper_utils import (train_classification_model_numpy,
                                 train_classification_model_pandas,
                                 train_regression_model_numpy,
                                 train_regression_model_pandas)
from wrapper_validator import validate_wrapped_tf_model


@pytest.mark.usefixtures('clean_dir')
class TestTensorflowModelWrapper(object):
    def test_wrap_keras_classification_model(self, iris):
        wrapped_init = wrapped_tensorflow_model_initializer(
            create_keras_classifier, model_task=ModelTask.CLASSIFICATION)
        train_classification_model_numpy(wrapped_init, iris)
        train_classification_model_pandas(wrapped_init, iris)

    def test_wrap_keras_regression_model(self, housing):
        wrapped_init = wrapped_tensorflow_model_initializer(
            create_keras_regressor, model_task=ModelTask.REGRESSION)
        train_regression_model_numpy(wrapped_init, housing)
        train_regression_model_pandas(wrapped_init, housing)

    def test_wrap_scikit_keras_regression_model(self, housing):
        wrapped_init = wrapped_tensorflow_model_initializer(
            create_scikit_keras_regressor, model_task=ModelTask.REGRESSION)
        train_regression_model_numpy(wrapped_init, housing)
        train_regression_model_pandas(wrapped_init, housing)


class TensorflowModelInitializer():
    def __init__(self, model_initializer, model_task):
        self._model_initializer = model_initializer
        self._model_task = model_task

    def __call__(self, X_train, y_train):
        fitted_model = self._model_initializer(X_train, y_train)
        wrapped_tf_model = WrappedTensorflowModel(fitted_model)
        validate_wrapped_tf_model(wrapped_tf_model, X_train, self._model_task)
        return wrapped_tf_model


def wrapped_tensorflow_model_initializer(model_initializer, model_task):
    return TensorflowModelInitializer(model_initializer, model_task)
