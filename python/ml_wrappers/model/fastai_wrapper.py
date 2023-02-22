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
        dl = self._model.dls[0]
        self.cat_cols = dl.dataset.cat_names
        self.cont_cols = dl.dataset.cont_names

    def _fastai_predict(self, dataset, index, model=None):
        """Predict the output using the wrapped FastAI model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :param index: The index into the predicted data.
            Index 1 is for the predicted class and index
            2 is for the predicted probability.
        :type index: int
        :param model: The model to use for prediction.
            If None, the wrapped model is used.
        :type model: fastai.learner.Learner
        :return: The predicted data.
        :rtype: numpy.ndarray
        """
        if model is None:
            model = self._model
        predictions = []
        for i in range(len(dataset)):
            row = dataset.iloc[i]
            # get only feature columns for prediction
            row = row[self.cat_cols + self.cont_cols]
            predictions.append(np.array(model.predict(row)[index]))
        predictions = np.array(predictions)
        if index == 1:
            is_boolean = predictions.dtype == bool
            if is_boolean:
                predictions = predictions.astype(int)
        return predictions

    def _fastai_predict_without_callbacks(self, dataset, index):
        """Predict the output using the wrapped FastAI model without callbacks.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :param index: The index into the predicted data.
            Index 1 is for the predicted class and index
            2 is for the predicted probability.
        :type index: int
        :return: The predicted data.
        :rtype: numpy.ndarray
        """
        removed_cbs = []
        default_cbs = ['TrainEvalCallback', 'Recorder', 'CastToTensor']
        for cb in self._model.cbs:
            if cb.__class__.__name__ not in default_cbs:
                removed_cbs.append(cb)
        with self._model.removed_cbs(removed_cbs) as model:
            return self._fastai_predict(dataset, index, model)

    def predict(self, dataset):
        """Predict the output value using the wrapped FastAI model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        try:
            return self._fastai_predict(dataset, 1)
        except Exception:
            return self._fastai_predict_without_callbacks(dataset, 1)

    def predict_proba(self, dataset):
        """Predict the output probability using the FastAI model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        try:
            return self._fastai_predict(dataset, 2)
        except Exception:
            return self._fastai_predict_without_callbacks(dataset, 2)
