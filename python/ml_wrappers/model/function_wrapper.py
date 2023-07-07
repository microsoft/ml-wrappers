# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines helper utilities for resolving prediction function shape inconsistencies."""

import logging

import numpy as np

from ..common.constants import ModelTask

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


try:
    import torch
    pytorch_installed = True
except ImportError:
    pytorch_installed = False
    module_logger.debug('Could not import torch, required if using a PyTorch model')


def _convert_to_two_cols(function, examples):
    """In classification case, convert the function's output to two columns if it outputs one column.

    :param function: The prediction function to evaluate on the examples.
    :type function: function
    :param examples: The model evaluation examples.
    :type examples: numpy.ndarray or list
    :return: The function chosen from given model and classification domain.
    :rtype: (function, str)
    """
    # Add wrapper function to convert output to 2D array, check values to decide on whether
    # to throw, or create two columns [1-p, p], or leave just one in multiclass one-class edge-case
    result = function(examples)
    single_row_result = function(examples[0:1])
    is_single_row_1d_result = len(single_row_result.shape) == 1
    # If the function gives a 2D array of one column, we will need to reshape it prior to concat
    is_2d_result = len(result.shape) == 2
    is_multi_row_multi_cols = result.shape[1] > 1
    reshape_1_row_1d = is_single_row_1d_result and is_2d_result and is_multi_row_multi_cols
    wrapper = _FunctionWrapper(function)
    if reshape_1_row_1d:
        # special weird case for multiclass KerasClassifier where single
        # row result is 1D but multi row result is 2D, even when passing
        # a single row as either a 1D or 2D array
        return (wrapper._function_2D_single_row_wrapper_2D_result, ModelTask.CLASSIFICATION)
    convert_to_two_cols = False
    for value in result:
        if value < 0 or value > 1:
            error = "Probability values outside of valid range: "
            error += str(value)
            raise Exception(error)
        if value < 1:
            convert_to_two_cols = True
    if convert_to_two_cols:
        # Create two cols, [1-p, p], from evaluation result
        if is_2d_result:
            return (wrapper._function_2D_two_cols_wrapper_2D_result, ModelTask.CLASSIFICATION)
        else:
            return (wrapper._function_2D_two_cols_wrapper_1D_result, ModelTask.CLASSIFICATION)
    else:
        if is_2d_result:
            return (function, ModelTask.CLASSIFICATION)
        else:
            return (wrapper._function_2D_one_col_wrapper, ModelTask.CLASSIFICATION)


class _FunctionWrapper(object):
    """Wraps a function to reshape the input and output data.

    :param function: The prediction function to evaluate on the examples.
    :type function: function
    """

    def __init__(self, function, base_dims=1, is_pytorch_image_model=False):
        """Wraps a function to reshape the input and output data.

        :param function: The prediction function to evaluate on the examples.
        :type function: function
        :param base_dims: The base dimensions of the input data.
        :type base_dims: int
        """
        self._function = function
        self._base_dims = base_dims
        self._is_pytorch_image_model = is_pytorch_image_model

    def _function_input_expand_wrapper(self, dataset):
        """Wraps a function that expands the dims of input dataset.

        Note unlike other functions, this runs on the input dataset type,
        and not the converted type.  Note in practice only numpy and tensor
        arrays have been seen to have this issue for models, this function
        does not apply to pandas dataframes.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray or torch.Tensor
        :return: A wrapped function.
        :rtype: function
        """
        if len(dataset.shape) == self._base_dims:
            is_tensor = False
            if pytorch_installed and isinstance(dataset, torch.Tensor):
                is_tensor = True
            # pytorch requires extra dimension for grayscale images
            # this only needs to be done if image is not yet in tensor format
            is_2d = len(dataset.shape) == 2
            if not is_tensor and self._is_pytorch_image_model and is_2d:
                dataset = np.expand_dims(dataset, axis=2)
            dataset = np.expand_dims(dataset, axis=0)
            if is_tensor:
                dataset = torch.tensor(dataset)
        return self._function(dataset)

    def _function_flatten(self, dataset):
        """Wraps a function that flattens the input dataset from 2D to 1D.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        return self._function(dataset).flatten()

    def _function_2D_two_cols_wrapper_2D_result(self, dataset):
        """Wraps a function that creates two columns, [1-p, p], from 2D array of one column evaluation result.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)[:, 0]
        return np.stack([1 - result, result], axis=-1)

    def _function_2D_two_cols_wrapper_1D_result(self, dataset):
        """Wraps a function that creates two columns, [1-p, p], from evaluation result that is a 1D array.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)
        return np.stack([1 - result, result], axis=-1)

    def _function_2D_one_col_wrapper(self, dataset):
        """Wraps a function that creates one column in rare edge case scenario for multiclass one-class result.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)
        return result.reshape(result.shape[0], 1)

    def _function_2D_single_row_wrapper_2D_result(self, dataset):
        """Wraps a single row result in rare multiclass case.

        In this rare multiclass case the single row result
        is 1D but multi row result is 2D.  The single row,
        even when formatted as a 1D or 2D array, gives similar
        consistent 1D array result, but as soon more than 2
        rows are given the result is a 2D array.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)
        if dataset.shape[0] == 1:
            return result.reshape(1, result.shape[0])
        return result


class _MultiVsSingleInstanceFunctionResolver(object):
    """Wraps function output to be same in single and multi-instance cases."""

    def __init__(self, function):
        """Wraps function output to be same in single and multi-instance cases

        :param function: The prediction function to evaluate on the examples.
        :type function: function
        """
        self._function = function

    def _add_single_predict_dimension(self, dataset):
        """Wraps function output for single instance case to add dimension.

        Ensures the single instance case returns same dimensionality
        as multi-instance prediction.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)
        if len(dataset.shape) == 1:
            return np.expand_dims(result, axis=0)
        else:
            return result
