# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for WrappedPytorchModel"""

import pytest
from common_utils import (create_pytorch_multiclass_classifier,
                          create_pytorch_regressor)
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.model import WrappedPytorchModel
from train_wrapper_utils import (train_classification_model_numpy,
                                 train_regression_model_numpy)
from wrapper_validator import validate_wrapped_pytorch_model


@pytest.mark.usefixtures('clean_dir')
class TestPytorchModelWrapper(object):
    def test_wrap_pytorch_classification_model(self, iris):
        wrapped_init = wrapped_pytorch_model_initializer(
            create_pytorch_multiclass_classifier,
            model_task=ModelTask.CLASSIFICATION)
        train_classification_model_numpy(wrapped_init, iris)
        train_classification_model_numpy(wrapped_init, iris,
                                         use_dataset_wrapper=False)

    def test_wrap_pytorch_regression_model(self, housing):
        wrapped_init = wrapped_pytorch_model_initializer(
            create_pytorch_regressor, model_task=ModelTask.REGRESSION)
        train_regression_model_numpy(
            wrapped_init, housing)


class PytorchModelInitializer():
    def __init__(self, model_initializer, model_task):
        self._model_initializer = model_initializer
        self._model_task = model_task

    def __call__(self, X_train, y_train):
        fitted_model = self._model_initializer(X_train, y_train)
        wrapped_pytorch_model = WrappedPytorchModel(fitted_model)
        validate_wrapped_pytorch_model(wrapped_pytorch_model, X_train,
                                       self._model_task)
        return wrapped_pytorch_model


def wrapped_pytorch_model_initializer(model_initializer, model_task):
    return PytorchModelInitializer(model_initializer, model_task)
