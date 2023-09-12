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

try:
    from torchvision.transforms import ToTensor
except ImportError:
    module_logger.debug('Could not import torchvision, required if using'
                        ' a vision PyTorch model')


class WrappedPytorchModel(object):
    """A class for wrapping a PyTorch model.

    Note at time of initialization, since we don't have
    access to the dataset, we can't infer if this is for
    classification or regression case.  Hence, we add
    the predict_classes method for classification, and keep
    predict for either outputting values in regression or
    probabilities in classification.
    """

    def __init__(self, model, image_to_tensor=False):
        """Initialize the WrappedPytorchModel with the model and evaluation function.

        :param model: The PyTorch model to wrap.
        :type model: torch.nn.Module
        :param image_to_tensor: Whether to convert the image to tensor.
        :type image_to_tensor: bool
        """
        self._model = model
        # Set eval automatically for user for batchnorm and dropout layers
        self._model.eval()
        self._image_to_tensor = image_to_tensor

    def _convert_to_tensor(self, dataset):
        """Convert the dataset to a pytorch tensor.

        For image datasets, we use ToTensor from torchvision,
        which moves channel to the first dimension and for
        2D images adds a third dimension.

        :param dataset: The dataset to convert.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The converted dataset.
        :rtype: torch.Tensor
        """
        # Convert the data to pytorch Variable
        if isinstance(dataset, pd.DataFrame):
            if self._image_to_tensor:
                dataset = dataset.iloc[0]
            dataset = dataset.values
            # If represented as a list of arrays,
            # convert to a 3D array instead of array
            # of 2D arrays
            if len(dataset.shape) == 1:
                if self._image_to_tensor and len(dataset[0].shape) == 2:
                    # add channel to end of image if 2D grayscale
                    for i in range(dataset.shape[0]):
                        dataset[i] = np.expand_dims(dataset[i], axis=2)
                dataset = np.stack(dataset)
        # If not already tensor, convert
        if not isinstance(dataset, torch.Tensor):
            if self._image_to_tensor:
                # For torchvision images, can only convert one
                # image at a time
                # Note pytorch wrapper expects extra dimension for rows
                # to be expanded in evaluator for image case,
                # otherwise this code won't work for a single
                # image input to predict call
                rows = []
                for row in range(dataset.shape[0]):
                    instance = dataset[row]
                    if not isinstance(instance, torch.Tensor):
                        instance = ToTensor()(instance)
                    rows.append(instance)
                dataset = torch.stack(rows)
            else:
                dataset = torch.Tensor(dataset)
        return dataset

    def predict(self, dataset):
        """Predict the output using the wrapped PyTorch model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The prediction results.
        :rtype: numpy.ndarray
        """
        wrapped_dataset = self._convert_to_tensor(dataset)
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
        wrapped_dataset = self._convert_to_tensor(dataset)
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
