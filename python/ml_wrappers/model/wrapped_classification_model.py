# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a class for wrapping classification models."""

import numpy as np
import pandas as pd
from ml_wrappers.common.constants import ModelTask, SKLearn
from ml_wrappers.model.base_wrapped_model import BaseWrappedModel
from ml_wrappers.model.function_wrapper import _FunctionWrapper
from ml_wrappers.model.pytorch_wrapper import WrappedPytorchModel
from ml_wrappers.model.tensorflow_wrapper import (WrappedTensorflowModel,
                                                  is_sequential)


class WrappedClassificationModel(BaseWrappedModel):
    """A class for wrapping a classification model."""

    def __init__(self, model, eval_function, examples=None):
        """Initialize the WrappedClassificationModel with the model and evaluation function."""
        super(WrappedClassificationModel, self).__init__(model, eval_function, examples, ModelTask.CLASSIFICATION)

    def predict(self, dataset):
        """Predict the output using the wrapped classification model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        is_tf_seq = is_sequential(self._model)
        is_wrapped_pytroch = isinstance(self._model, WrappedPytorchModel)
        is_wrapped_tf = isinstance(self._model, WrappedTensorflowModel)
        if is_tf_seq or is_wrapped_pytroch or is_wrapped_tf:
            wrapped_predict_classes = self._wrap_function(self._model.predict_classes)
            return wrapped_predict_classes(dataset).flatten()
        wrapped_predict = self._wrap_function(self._model.predict)
        preds = wrapped_predict(dataset)
        if isinstance(preds, pd.DataFrame):
            preds = preds.values.ravel()
        # Handle possible case where the model has only a predict function and it outputs probabilities
        # Note this is different from WrappedClassificationWithoutProbaModel where there is no predict_proba
        # method but the predict method outputs classes
        has_predict_proba = hasattr(self._model, SKLearn.PREDICT_PROBA)
        if not has_predict_proba:
            if len(preds.shape) == 1:
                return np.argmax(preds)
            else:
                return np.argmax(preds, axis=1)
        # Handle the case that the model predicts a two-dimensional array of one column
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        return np.array(preds)

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        proba_preds = self._eval_function(dataset)
        if isinstance(proba_preds, pd.DataFrame):
            proba_preds = proba_preds.values

        return proba_preds

    def _wrap_function(self, function):
        """Wrap a function to conform to the prediction input contracts.

        If model requires _function_input_expand_wrapper, re-wraps
        the given function with _function_input_expand_wrapper.

        :param function: The function to wrap.
        :type function: function
        :return: The wrapped function.
        :rtype: function
        """
        eval_function = self._eval_function
        exp_wrapper = _FunctionWrapper._function_input_expand_wrapper
        exp_wrapper_name = exp_wrapper.__name__
        if eval_function.__name__ == exp_wrapper_name:
            base_dims = eval_function.__self__._base_dims
            function_wrapper = _FunctionWrapper(function, base_dims)
            function = function_wrapper._function_input_expand_wrapper
        return function
