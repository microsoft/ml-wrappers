# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a class for wrapping regression models."""

import pandas as pd
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.model.base_wrapped_model import BaseWrappedModel


class WrappedRegressionModel(BaseWrappedModel):
    """A class for wrapping a regression model."""

    def __init__(self, model, eval_function, examples=None):
        """Initialize the WrappedRegressionModel with the model and evaluation function."""
        super(WrappedRegressionModel, self).__init__(
            model, eval_function, examples, ModelTask.REGRESSION)

    def predict(self, dataset):
        """Predict the output using the wrapped regression model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        preds = self._eval_function(dataset)
        if isinstance(preds, pd.DataFrame):
            preds = preds.values.ravel()

        return preds
