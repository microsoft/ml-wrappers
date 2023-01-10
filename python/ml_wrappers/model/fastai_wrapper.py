# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines model wrappers and utilities for fastai tabular models."""

import numpy as np

FASTAI_TABULAR_MODEL_SUFFIX = "fastai.tabular.learner.TabularLearner'>"


def _is_fastai_tabular_model(model):
    """Check if the model is a fastai tabular model.

    :param model: The model to check.
    :type model: object
    :return: True if the model is a fastai model, False otherwise.
    :rtype: bool
    """
    return str(type(model)).endswith(FASTAI_TABULAR_MODEL_SUFFIX)


class WrappedFastAITabularModel(object):
    """A class for wrapping a FastAI tabular model in the scikit-learn style."""

    def __init__(self, model):
        """Initialize the WrappedFastAITabularModel.

        :param model: The model to wrap.
        :type model: fastai.learner.Learner
        """
        self._model = model

    def _fastai_predict(self, dataset, index):
        """Predict the output using the wrapped FastAI model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :param index: The index into the predicted data.
            Index 1 is for the predicted class and index
            2 is for the predicted probability.
        :type index: int
        :return: The predicted data.
        :rtype: numpy.ndarray
        """
        predictions = []
        for i in range(len(dataset)):
            row = dataset.iloc[i]
            predictions.append(np.array(self._model.predict(row)[index]))
        predictions = np.array(predictions)
        if index == 1:
            is_boolean = predictions.dtype == bool
            if is_boolean:
                predictions = predictions.astype(int)
        return predictions

    def predict(self, dataset):
        """Predict the output value using the wrapped FastAI model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        return self._fastai_predict(dataset, 1)

    def predict_proba(self, dataset):
        """Predict the output probability using the FastAI model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        return self._fastai_predict(dataset, 2)
