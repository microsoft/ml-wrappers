# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines helpful model wrapper and utils for implicitly rewrapping the model to conform to explainer contracts."""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from ..common.constants import (ModelTask, SKLearn, image_model_tasks,
                                text_model_tasks)
from ..dataset.dataset_wrapper import DatasetWrapper
from .function_wrapper import (_convert_to_two_cols, _FunctionWrapper,
                               _MultiVsSingleInstanceFunctionResolver)
from .image_model_wrapper import _wrap_image_model
from .pytorch_wrapper import WrappedPytorchModel
from .tensorflow_wrapper import WrappedTensorflowModel, is_sequential
from .text_model_wrapper import _is_transformers_pipeline, _wrap_text_model

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


try:
    import torch.nn as nn
except ImportError:
    module_logger.debug('Could not import torch, required if using a PyTorch model')


class BaseWrappedModel(object):
    """A base class for WrappedClassificationModel and WrappedRegressionModel."""

    def __init__(self, model, eval_function, examples, model_task):
        """Initialize the WrappedClassificationModel with the model and evaluation function."""
        self._eval_function = eval_function
        self._model = model
        self._examples = examples
        self._model_task = model_task

    def __getstate__(self):
        """Influence how BaseWrappedModel is pickled.

        Removes _eval_function which may not be serializable.

        :return state: The state to be pickled, with _eval_function removed.
        :rtype: dict
        """
        odict = self.__dict__.copy()
        if self._examples is not None:
            del odict['_eval_function']
        return odict

    def __setstate__(self, state):
        """Influence how BaseWrappedModel is unpickled.

        Re-adds _eval_function which may not be serializable.

        :param dict: A dictionary of deserialized state.
        :type dict: dict
        """
        self.__dict__.update(state)
        if self._examples is not None:
            eval_function, _ = _eval_model(self._model, self._examples, self._model_task)
            self._eval_function = eval_function


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
            return self._model.predict_classes(dataset).flatten()
        preds = self._model.predict(dataset)
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
        return preds

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        proba_preds = self._eval_function(dataset)
        if isinstance(proba_preds, pd.DataFrame):
            proba_preds = proba_preds.values

        return proba_preds


class WrappedRegressionModel(BaseWrappedModel):
    """A class for wrapping a regression model."""

    def __init__(self, model, eval_function, examples=None):
        """Initialize the WrappedRegressionModel with the model and evaluation function."""
        super(WrappedRegressionModel, self).__init__(model, eval_function, examples, ModelTask.REGRESSION)

    def predict(self, dataset):
        """Predict the output using the wrapped regression model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        preds = self._eval_function(dataset)
        if isinstance(preds, pd.DataFrame):
            preds = preds.values.ravel()

        return preds


class WrappedClassificationWithoutProbaModel(object):
    """A class for wrapping a classifier without a predict_proba method.

    Note: the classifier may not output numeric values for its predictions.
    We generate a trival boolean version of predict_proba
    """

    def __init__(self, model):
        """Initialize the WrappedClassificationWithoutProbaModel with the model."""
        self._model = model
        # Create a map from classes to index
        self._classes_to_index = {}
        for index, i in enumerate(self._model.classes_):
            self._classes_to_index[i] = index
        self._num_classes = len(self._model.classes_)

    def predict(self, dataset):
        """Predict the output using the wrapped regression model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        return self._model.predict(dataset)

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        predictions = self.predict(dataset)
        # Generate trivial boolean array for predictions
        probabilities = np.zeros((predictions.shape[0], self._num_classes))
        for row_idx, pred_class in enumerate(predictions):
            class_index = self._classes_to_index[pred_class]
            probabilities[row_idx, class_index] = 1
        return probabilities


def wrap_model(model, examples, model_task=ModelTask.UNKNOWN):
    """If needed, wraps the model in a common API based on model task and prediction function contract.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function.
    :param examples: The model evaluation examples.
        Note the examples will be wrapped in a DatasetWrapper, if not
        wrapped when input.
    :type examples: ml_wrappers.DatasetWrapper or numpy.ndarray
        or pandas.DataFrame or panads.Series or scipy.sparse.csr_matrix
        or shap.DenseData or torch.Tensor
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :return: The wrapper model.
    :rtype: model
    """
    if model_task == ModelTask.UNKNOWN and _is_transformers_pipeline(model):
        # TODO: can we also dynamically figure out the task if it was
        # originally unknown for text scenarios?
        raise ValueError("ModelTask must be specified for text-based models")
    if model_task in text_model_tasks:
        return _wrap_text_model(model, examples, model_task, False)[0]
    if model_task in image_model_tasks:
        return _wrap_image_model(model, examples, model_task, False)[0]
    return _wrap_model(model, examples, model_task, False)[0]


def _wrap_model(model, examples, model_task, is_function):
    """If needed, wraps the model or function in a common API based on model task and prediction function contract.

    :param model: The model or function to evaluate on the examples.
    :type model: function or model with a predict or predict_proba function
    :param examples: The model evaluation examples.
        Note the examples will be wrapped in a DatasetWrapper, if not
        wrapped when input.
    :type examples: ml_wrappers.DatasetWrapper or numpy.ndarray
        or pandas.DataFrame or panads.Series or scipy.sparse.csr_matrix
        or shap.DenseData or torch.Tensor
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :return: The function chosen from given model and chosen domain, or model wrapping the function and chosen domain.
    :rtype: (function, str) or (model, str)
    """
    if not isinstance(examples, DatasetWrapper):
        examples = DatasetWrapper(examples)
    if is_function:
        return _eval_function(model, examples, model_task)
    else:
        try:
            if isinstance(model, nn.Module):
                # Wrap the model in an extra layer that converts the numpy array
                # to pytorch Variable and adds predict and predict_proba functions
                model = WrappedPytorchModel(model)
        except (NameError, AttributeError):
            module_logger.debug('Could not import torch, required if using a pytorch model')
        if is_sequential(model):
            model = WrappedTensorflowModel(model)
        if _classifier_without_proba(model):
            model = WrappedClassificationWithoutProbaModel(model)
        eval_function, eval_ml_domain = _eval_model(model, examples, model_task)
        if eval_ml_domain == ModelTask.CLASSIFICATION:
            return WrappedClassificationModel(model, eval_function, examples), eval_ml_domain
        else:
            return WrappedRegressionModel(model, eval_function, examples), eval_ml_domain


def _classifier_without_proba(model):
    """Returns True if the given model is a classifier without predict_proba, eg SGDClassifier.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function
    :return: True if the given model is a classifier without predict_proba.
    :rtype: bool
    """
    return isinstance(model, SGDClassifier) and not hasattr(model, SKLearn.PREDICT_PROBA)


def _eval_model(model, examples, model_task):
    """Return function from model and specify the ML Domain using model evaluation on examples.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function
    :param examples: The model evaluation examples.
    :type examples: ml_wrappers.DatasetWrapper
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :return: The function chosen from given model and chosen domain.
    :rtype: (function, str)
    """
    # TODO: Add more model types here
    is_tf_seq = is_sequential(model)
    is_wrapped_pytroch = isinstance(model, WrappedPytorchModel)
    is_wrapped_tf = isinstance(model, WrappedTensorflowModel)
    if is_tf_seq or is_wrapped_pytroch or is_wrapped_tf:
        if model_task == ModelTask.REGRESSION:
            return _eval_function(model.predict, examples, ModelTask.REGRESSION)
        result = model.predict_proba(examples.typed_wrapper_func(examples.dataset[0:1]))
        if result.shape[1] == 1 and model_task == ModelTask.UNKNOWN:
            raise Exception("Please specify model_task to disambiguate model type since "
                            "result of calling function is 2D array of one column.")
        else:
            return _eval_function(model.predict_proba, examples, ModelTask.CLASSIFICATION)
    else:
        has_predict_proba = hasattr(model, SKLearn.PREDICT_PROBA)
        # Note: Allow user to override default to use predict method for regressor
        if has_predict_proba and model_task != ModelTask.REGRESSION:
            return _eval_function(model.predict_proba, examples, model_task)
        else:
            return _eval_function(model.predict, examples, model_task)


def _eval_function(function, examples, model_task, wrapped=False):
    """Return function and specify the ML Domain using function evaluation on examples.

    :param function: The prediction function to evaluate on the examples.
    :type function: function
    :param examples: The model evaluation examples.
    :type examples: ml_wrappers.DatasetWrapper
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :param wrapped: Indicates if function has already been wrapped.
    :type wrapped: bool
    :return: The function chosen from given model and chosen domain.
    :rtype: (function, str)
    """
    # Try to run the function on a single example - if it doesn't work wrap
    # it in a function that converts a 1D array to 2D for those functions
    # that only support 2D arrays as input
    examples_dataset = examples.dataset
    if str(type(examples_dataset)).endswith(".DenseData'>"):
        examples_dataset = examples_dataset.data
    try:
        inst_result = function(examples.typed_wrapper_func(examples_dataset[0]))
        if inst_result is None:
            raise Exception("Wrapped function returned None in model wrapper when called on dataset")
        multi_inst_result = function(examples.typed_wrapper_func(examples_dataset[0:1]))
        if multi_inst_result.shape != inst_result.shape:
            if len(multi_inst_result.shape) == len(inst_result.shape) + 1:
                resolver = _MultiVsSingleInstanceFunctionResolver(function)
                return _eval_function(resolver._add_single_predict_dimension, examples, model_task)
            else:
                raise Exception("Wrapped function dimensions for single and multi predict unresolvable")
    except Exception as ex:
        # If function has already been wrapped, re-throw error to prevent stack overflow
        if wrapped:
            raise ex
        wrapper = _FunctionWrapper(function)
        return _eval_function(wrapper._function_input_1D_wrapper, examples, model_task, wrapped=True)
    if len(inst_result.shape) == 2:
        # If the result of evaluation the function is a 2D array of 1 column,
        # and they did not specify classifier or regressor, throw exception
        # to force the user to disambiguate the results.
        if inst_result.shape[1] == 1:
            if model_task == ModelTask.UNKNOWN:
                if isinstance(inst_result, pd.DataFrame):
                    return (function, ModelTask.REGRESSION)
                raise Exception("Please specify model_task to disambiguate model type since "
                                "result of calling function is 2D array of one column.")
            elif model_task == ModelTask.CLASSIFICATION:
                return _convert_to_two_cols(function, examples_dataset)
            else:
                # model_task == ModelTask.REGRESSION
                # In case user specified a regressor but we have a 2D output with one column,
                # we want to flatten the function to 1D
                wrapper = _FunctionWrapper(function)
                return (wrapper._function_flatten, model_task)
        else:
            if model_task == ModelTask.UNKNOWN or model_task == ModelTask.CLASSIFICATION:
                return (function, ModelTask.CLASSIFICATION)
            else:
                raise Exception("Invalid shape for prediction: "
                                "Regression function cannot output 2D array with multiple columns")
    elif len(inst_result.shape) == 1:
        if model_task == ModelTask.UNKNOWN:
            return (function, ModelTask.REGRESSION)
        elif model_task == ModelTask.CLASSIFICATION:
            return _convert_to_two_cols(function, examples_dataset)
        return (function, model_task)
    elif len(inst_result.shape) == 0:
        # single value returned, flatten output array
        wrapper = _FunctionWrapper(function)
        return (wrapper._function_flatten, model_task)
    raise Exception("Failed to wrap function, may require custom wrapper for input function or model")
