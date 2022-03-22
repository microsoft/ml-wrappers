# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines wrappers for text-based models."""

import numpy as np

from ..common.warnings_suppressor import shap_warnings_suppressor

with shap_warnings_suppressor():
    try:
        from shap import models
        from shap.utils import safe_isinstance
        shap_installed = True
    except BaseException:
        shap_installed = False


def _is_transformers_pipeline(model):
    return shap_installed and safe_isinstance(
        model, "transformers.pipelines.Pipeline")


def _wrap_text_model(model, examples, model_task, is_function):
    """If needed, wraps the model or function in a common API.

    Wraps the model based on model task and prediction function contract.

    :param model: The model or function to evaluate on the examples.
    :type model: function or model with a predict or predict_proba function
    :param examples: The model evaluation examples.
        Note the examples will be wrapped in a DatasetWrapper, if not
        wrapped when input.
    :type examples: ml_wrappers.DatasetWrapper or numpy.ndarray
        or pandas.DataFrame or panads.Series or scipy.sparse.csr_matrix
        or shap.DenseData or torch.Tensor
    :param model_task: Parameter to specify whether the model is a
        'text_classification', 'sentiment_analysis', 'question_answering',
        'entailment' or 'summarizations' text model.
    :type model_task: str
    :return: The function chosen from given model and chosen domain, or
        model wrapping the function and chosen domain.
    :rtype: (function, str) or (model, str)
    """
    if _is_transformers_pipeline(model):
        _wrapped_model = WrappedTransformersModel(model)
    else:
        _wrapped_model = model
    return _wrapped_model, model_task


class WrappedTransformersModel(object):
    """A class for wrapping a Transformers model in the scikit-learn style."""

    def __init__(self, model):
        """Initialize the WrappedTransformersModel with the model."""
        self._model = model
        self._wrapped_model = models.TransformersPipeline(model)

    def predict(self, dataset):
        """Predict the output using the wrapped Transformers model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        pipeline_dicts = self._wrapped_model.inner_model(dataset)
        output = []
        for val in pipeline_dicts:
            if not isinstance(val, list):
                val = [val]
            scores = [obj["score"] for obj in val]
            max_score_index = np.argmax(scores)
            output.append(max_score_index)
        return np.array(output)

    def predict_proba(self, dataset):
        """Predict the output probability using the Transformers model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        return self._wrapped_model(dataset)
