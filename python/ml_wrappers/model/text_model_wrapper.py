# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines wrappers for text-based models."""

import numpy as np
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.common.warnings_suppressor import shap_warnings_suppressor
from ml_wrappers.model.model_utils import (MULTILABEL_THRESHOLD,
                                           _is_transformers_pipeline)

with shap_warnings_suppressor():
    try:
        from shap import models
        shap_installed = True
    except BaseException:
        shap_installed = False


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
    _wrapped_model = model
    if _is_transformers_pipeline(model):
        if model_task == ModelTask.TEXT_CLASSIFICATION:
            _wrapped_model = WrappedTextClassificationModel(model)
        elif model_task == ModelTask.QUESTION_ANSWERING:
            _wrapped_model = WrappedQuestionAnsweringModel(model)
        elif model_task == ModelTask.MULTILABEL_TEXT_CLASSIFICATION:
            _wrapped_model = WrappedTextClassificationModel(model, multilabel=True)
    return _wrapped_model, model_task


class WrappedTextClassificationModel(object):
    """A class for wrapping a Transformers model in the scikit-learn style."""

    def __init__(self, model, multilabel=False):
        """Initialize the WrappedTextClassificationModel."""
        self._model = model
        if not shap_installed:
            raise ImportError("SHAP is not installed. Please install it " +
                              "to use WrappedTextClassificationModel.")
        self._wrapped_model = models.TransformersPipeline(model)
        self._multilabel = multilabel

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
            if self._multilabel:
                threshold = MULTILABEL_THRESHOLD
                # jagged, thresholded array of labels model predicted
                labels = np.where(np.array(scores) > threshold)
                predictions = np.zeros(len(scores))
                # indicator matrix of labels since numpy does not
                # support jagged arrays, which seems to be the format
                # scikit-learn MultiOutputClassifier uses,
                # see sklearn.multioutput.MultiOutputClassifier.predict
                predictions[labels] = 1
                output.append(predictions)
            else:
                max_score_index = np.argmax(scores)
                output.append(max_score_index)
        return np.array(output)

    def predict_proba(self, dataset):
        """Predict the output probability using the Transformers model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        return self._wrapped_model(dataset)


class WrappedQuestionAnsweringModel(object):
    """A class for wrapping a Transformers model in the scikit-learn style."""

    def __init__(self, model):
        """Initialize the WrappedQuestionAnsweringModel."""
        self._model = model

    def predict(self, dataset):
        """Predict the output using the wrapped Transformers model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        """
        output = []
        for context, question in zip(dataset['context'], dataset['questions']):
            answer = self._model({'context': context, 'question': question})
            output.append(answer['answer'])
        return output
