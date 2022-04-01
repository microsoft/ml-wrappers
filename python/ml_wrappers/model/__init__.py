# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Common infrastructure, class hierarchy and utilities for model explanations."""

from .model_wrapper import (WrappedClassificationModel, WrappedRegressionModel,
                            _wrap_model, wrap_model)
from .pytorch_wrapper import WrappedPytorchModel
from .tensorflow_wrapper import WrappedTensorflowModel, is_sequential

__all__ = ['WrappedClassificationModel', 'WrappedPytorchModel',
           'WrappedRegressionModel', 'WrappedTensorflowModel',
           '_wrap_model', 'is_sequential', 'wrap_model']
