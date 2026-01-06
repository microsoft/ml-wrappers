# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for WrappedTensorflowModel"""

import sys

import pytest
import tensorflow as tf
from common_utils import (create_keras_classifier, create_keras_regressor,
                          create_scikit_keras_regressor)
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.model import WrappedTensorflowModel
from ml_wrappers.model.tensorflow_wrapper import is_sequential
from train_wrapper_utils import (train_classification_model_numpy,
                                 train_classification_model_pandas,
                                 train_regression_model_numpy,
                                 train_regression_model_pandas)
from wrapper_validator import validate_wrapped_tf_model


@pytest.mark.usefixtures('_clean_dir')
class TestTensorflowModelWrapper(object):
    # Skip on macOS due to TensorFlow/Keras hanging and compatibility issues
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='TensorFlow/Keras hangs on macOS')
    def test_wrap_keras_classification_model(self, iris):
        wrapped_init = wrapped_tensorflow_model_initializer(
            create_keras_classifier, model_task=ModelTask.CLASSIFICATION)
        train_classification_model_numpy(wrapped_init, iris)
        train_classification_model_pandas(wrapped_init, iris)

    # Skip on macOS due to TensorFlow/Keras hanging and compatibility issues
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='TensorFlow/Keras hangs on macOS')
    def test_wrap_keras_regression_model(self, housing):
        wrapped_init = wrapped_tensorflow_model_initializer(
            create_keras_regressor, model_task=ModelTask.REGRESSION)
        train_regression_model_numpy(wrapped_init, housing)
        train_regression_model_pandas(wrapped_init, housing)

    # Skip on macOS due to TensorFlow/Keras hanging and compatibility issues
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='TensorFlow/Keras hangs on macOS')
    def test_wrap_scikit_keras_regression_model(self, housing):
        wrapped_init = wrapped_tensorflow_model_initializer(
            create_scikit_keras_regressor, model_task=ModelTask.REGRESSION)
        train_regression_model_numpy(wrapped_init, housing)
        train_regression_model_pandas(wrapped_init, housing)

    def test_validate_is_sequential(self):
        sequential_layer = tf.keras.Sequential(layers=None, name=None)
        assert is_sequential(sequential_layer)


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
