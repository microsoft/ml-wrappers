# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines helper utilities for resolving prediction function shape inconsistencies."""

import numpy as np

from ..common.constants import ModelTask


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
    # If the function gives a 2D array of one column, we will need to reshape it prior to concat
    is_2d_result = len(result.shape) == 2
    convert_to_two_cols = False
    for value in result:
        if value < 0 or value > 1:
            raise Exception("Probability values outside of valid range: " + str(value))
        if value < 1:
            convert_to_two_cols = True
    wrapper = _FunctionWrapper(function)
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

    def __init__(self, function):
        """Wraps a function to reshape the input and output data.

        :param function: The prediction function to evaluate on the examples.
        :type function: function
        """
        self._function = function

    def _function_input_1D_wrapper(self, dataset):
        """Wraps a function that reshapes the input dataset to be 2D from 1D.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.ndarray
        :return: A wrapped function.
        :rtype: function
        """
        if len(dataset.shape) == 1:
            dataset = dataset.reshape(1, -1)
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
