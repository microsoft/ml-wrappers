# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines model wrappers and utilities for pytorch models."""

import logging

import numpy as np
import pandas as pd

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


try:
    import torch
except ImportError:
    module_logger.debug('Could not import torch, required if using a PyTorch model')


class WrappedPytorchModel(object):
    """A class for wrapping a PyTorch model.

    Note at time of initialization, since we don't have
    access to the dataset, we can't infer if this is for
    classification or regression case.  Hence, we add
    the predict_classes method for classification, and keep
    predict for either outputting values in regression or
    probabilities in classification.
    """

    def __init__(self, model):
        """Initialize the WrappedPytorchModel with the model and evaluation function.

        :param model: The PyTorch model to wrap.
        :type model: torch.nn.Module
        """
        self._model = model
        # Set eval automatically for user for batchnorm and dropout layers
        self._model.eval()

    def predict(self, dataset):
        """Predict the output using the wrapped PyTorch model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The prediction results.
        :rtype: numpy.ndarray
        """
        # Convert the data to pytorch Variable
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        wrapped_dataset = torch.Tensor(dataset)
        with torch.no_grad():
            result = self._model(wrapped_dataset).numpy()
        # Reshape to 2D if output is 1D and input has one row
        if len(dataset.shape) == 1:
            result = result.reshape(1, -1)
        return result

    def predict_classes(self, dataset):
        """Predict the class using the wrapped PyTorch model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted classes.
        :rtype: numpy.ndarray
        """
        # Convert the data to pytorch Variable
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        wrapped_dataset = torch.Tensor(dataset)
        with torch.no_grad():
            result = self._model(wrapped_dataset)
        result_len = len(result.shape)
        if result_len == 1 or (result_len > 1 and result.shape[1] == 1):
            result = np.where(result.numpy() > 0.5, 1, 0)
        else:
            result = torch.max(result, 1)[1].numpy()
        # Reshape to 2D if output is 1D and input has one row
        if len(dataset.shape) == 1:
            result = result.reshape(1, -1)
        return result

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped PyTorch model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        return self.predict(dataset)
