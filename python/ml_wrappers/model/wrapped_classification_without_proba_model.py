# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a class for wrapping classifiers without predict_proba."""

import numpy as np


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
