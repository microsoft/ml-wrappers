# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines model wrappers and utilities for tensorflow models."""

import numpy as np
import pandas as pd

PREDICT_CLASSES = 'predict_classes'


def is_sequential(model):
    """Returns True if the model is a sequential model.

    Note the model class name can either be
    keras.engine.sequential.Sequential or
    tensorflow.python.keras.engine.sequential.Sequential
    depending on the tensorflow version.
    The check should include both of these cases.

    :param model: The model to check.
    :type model: tf.keras.Model
    :return: True if the model is a sequential model.
    :rtype: bool
    """
    return str(type(model)).endswith("keras.engine.sequential.Sequential'>")


class WrappedTensorflowModel(object):
    """A class for wrapping a TensorFlow model.

    Note at time of initialization, since we don't have
    access to the dataset, we can't infer if this is for
    classification or regression case.  Hence, we add
    the predict_classes method for classification, and keep
    predict for either outputting values in regression or
    probabilities in classification.
    """

    def __init__(self, model):
        """Initialize the WrappedTensorflowModel with the model.

        :param model: The model to wrap.
        :type model: tf.keras.Model
        """
        self._model = model

    def predict(self, dataset):
        """Predict the output using the wrapped TensorFlow model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The prediction results.
        :rtype: numpy.ndarray
        """
        # Convert the data to numpy
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        return self._model.predict(dataset)

    def predict_classes(self, dataset):
        """Predict the class using the wrapped TensorFlow model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted classes.
        :rtype: numpy.ndarray
        """
        # Note predict_classes was removed for models after
        # tensorflow version 2.6
        if hasattr(self._model, PREDICT_CLASSES):
            return self._model.predict_classes(dataset)
        probabilities = self.predict_proba(dataset)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped TensorFlow model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        return self.predict(dataset)
