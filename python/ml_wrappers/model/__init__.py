# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Common infrastructure, class hierarchy and utilities for model explanations."""

from .model_wrapper import WrappedPytorchModel, _wrap_model, wrap_model

__all__ = ['_wrap_model', 'wrap_model', 'WrappedPytorchModel']
