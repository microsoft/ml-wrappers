# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pandas as pd

from ..common.constants import ModelTask, SKLearn
from .function_wrapper import (_convert_to_two_cols, _FunctionWrapper,
                               _MultiVsSingleInstanceFunctionResolver)
from .pytorch_wrapper import WrappedPytorchModel
from .tensorflow_wrapper import WrappedTensorflowModel, is_sequential


def _is_classification_task(task):
    """Return True if the task is a classification task.

    :param task: The task to check.
    :type task: str
    :return: True if the task is a classification task.
    :rtype: bool
    """
    return task == ModelTask.CLASSIFICATION or task == ModelTask.IMAGE_CLASSIFICATION


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
    is_tf_seq = is_sequential(model)
    is_wrapped_pytorch = isinstance(model, WrappedPytorchModel)
    is_wrapped_tf = isinstance(model, WrappedTensorflowModel)
    if is_tf_seq or is_wrapped_pytorch or is_wrapped_tf:
        if model_task == ModelTask.REGRESSION:
            return _eval_function(model.predict, examples, ModelTask.REGRESSION)
        if model_task == ModelTask.IMAGE_CLASSIFICATION:
            examples_dataset = examples.dataset
            if isinstance(examples_dataset, pd.DataFrame):
                return _eval_function(model.predict_proba, examples,
                                      model_task, wrapped=True)
            is_pytorch_image_model = True
            wrapper = _FunctionWrapper(model.predict_proba,
                                       len(examples_dataset[0].shape),
                                       is_pytorch_image_model)
            return _eval_function(wrapper._function_input_expand_wrapper,
                                  examples, model_task, wrapped=True)
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
        wrapper = _FunctionWrapper(function, len(examples_dataset[0].shape))
        return _eval_function(wrapper._function_input_expand_wrapper, examples, model_task, wrapped=True)
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
            elif _is_classification_task(model_task):
                return _convert_to_two_cols(function, examples_dataset)
            else:
                # model_task == ModelTask.REGRESSION
                # In case user specified a regressor but we have a 2D output with one column,
                # we want to flatten the function to 1D
                wrapper = _FunctionWrapper(function)
                return (wrapper._function_flatten, model_task)
        else:
            if model_task == ModelTask.UNKNOWN or _is_classification_task(model_task):
                return (function, ModelTask.CLASSIFICATION)
            else:
                raise Exception("Invalid shape for prediction: "
                                "Regression function cannot output 2D array with multiple columns")
    elif len(inst_result.shape) == 1:
        if model_task == ModelTask.UNKNOWN:
            return (function, ModelTask.REGRESSION)
        elif _is_classification_task(model_task):
            return _convert_to_two_cols(function, examples_dataset)
        return (function, model_task)
    elif len(inst_result.shape) == 0:
        # single value returned, flatten output array
        wrapper = _FunctionWrapper(function)
        return (wrapper._function_flatten, model_task)
    raise Exception("Failed to wrap function, may require custom wrapper for input function or model")
