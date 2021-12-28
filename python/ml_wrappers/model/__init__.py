# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Common infrastructure, class hierarchy and utilities for model explanations."""

from .model_wrapper import _wrap_model, wrap_model, WrappedPytorchModel

__all__ = ['_wrap_model', 'wrap_model', 'WrappedPytorchModel']
