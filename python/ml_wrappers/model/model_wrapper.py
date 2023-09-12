# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines helpful model wrapper and utils for implicitly rewrapping the model
    to conform to explainer contracts."""

import logging
import warnings
from typing import Union

import numpy as np
from ml_wrappers.model.wrapped_classification_model import \
    WrappedClassificationModel
from ml_wrappers.model.wrapped_classification_without_proba_model import \
    WrappedClassificationWithoutProbaModel
from ml_wrappers.model.wrapped_regression_model import WrappedRegressionModel
from sklearn.linear_model import SGDClassifier

from ..common.constants import (Device, ModelTask, SKLearn, image_model_tasks,
                                text_model_tasks)
from ..dataset.dataset_wrapper import DatasetWrapper
from .evaluator import _eval_function, _eval_model
from .fastai_wrapper import WrappedFastAITabularModel, _is_fastai_tabular_model
from .image_model_wrapper import _wrap_image_model
from .pytorch_wrapper import WrappedPytorchModel
from .tensorflow_wrapper import WrappedTensorflowModel, is_sequential
from .text_model_wrapper import _is_transformers_pipeline, _wrap_text_model

with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore', 'Starting from version 2.2.1', UserWarning)


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


try:
    import torch.nn as nn
except ImportError:
    module_logger.debug(
        'Could not import torch, required if using a PyTorch model')


def wrap_model(model, examples, model_task: str = ModelTask.UNKNOWN,
               num_classes: int = None, classes: Union[list, np.array] = None,
               device=Device.AUTO.value):
    """If needed, wraps the model in a common API based on model task and
        prediction function contract.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function.
    :param examples: The model evaluation examples.
        Note the examples will be wrapped in a DatasetWrapper, if not
        wrapped when input.
    :type examples: ml_wrappers.DatasetWrapper or numpy.ndarray
        or pandas.DataFrame or panads.Series or scipy.sparse.csr_matrix
        or shap.DenseData or torch.Tensor
    :param model_task: Optional parameter to specify whether the model
                       is a classification or regression model.
                       In most cases, the type of the model can be inferred
                       based on the shape of the output, where a classifier
                       has a predict_proba method and outputs a 2 dimensional
                       array, while a regressor has a predict method and
                       outputs a 1 dimensional array.
    :param classes: optional parameter specifying a list of class names
        the dataset
    :type classes: list or np.ndarray
    :param num_classes: optional parameter specifying the number of classes in
        the dataset
    :type num_classes: int
    :type model_task: str
    :param device: optional parameter specifying the device to move the model
        to. If not specified, then cpu is the default
    :type device: str, for instance: 'cpu', 'cuda'
    :return: The wrapper model.
    :rtype: model
    """
    if model_task == ModelTask.UNKNOWN and _is_transformers_pipeline(model):
        # TODO: can we also dynamically figure out the task if it was
        # originally unknown for text scenarios?
        raise ValueError("ModelTask must be specified for text-based models")
    if model_task in text_model_tasks:
        return _wrap_text_model(model, examples, model_task, False)[0]
    if model_task in image_model_tasks:
        return _wrap_image_model(model, examples, model_task,
                                 False, num_classes, classes,
                                 device)[0]
    return _wrap_model(model, examples, model_task, False)[0]


def _wrap_model(model, examples, model_task, is_function):
    """If needed, wraps the model or function in a common API based on model
        task and prediction function contract.

    :param model: The model or function to evaluate on the examples.
    :type model: function or model with a predict or predict_proba function
    :param examples: The model evaluation examples.
        Note the examples will be wrapped in a DatasetWrapper, if not
        wrapped when input.
    :type examples: ml_wrappers.DatasetWrapper or numpy.ndarray
        or pandas.DataFrame or panads.Series or scipy.sparse.csr_matrix
        or shap.DenseData or torch.Tensor
    :param model_task: Optional parameter to specify whether the model
                       is a classification or regression model.
                       In most cases, the type of the model can be inferred
                       based on the shape of the output, where a classifier
                       has a predict_proba method and outputs a 2 dimensional
                       array, while a regressor has a predict method and
                       outputs a 1 dimensional array.
    :type model_task: str
    :return: The function chosen from given model and chosen domain, or
             model wrapping the function and chosen domain.
    :rtype: (function, str) or (model, str)
    """
    if not isinstance(examples, DatasetWrapper):
        examples = DatasetWrapper(examples)
    if is_function:
        return _eval_function(model, examples, model_task)
    else:
        try:
            if isinstance(model, nn.Module):
                # Wrap the model in an extra layer that
                # converts the numpy array

                # to pytorch Variable and adds predict and
                # predict_proba functions
                model = WrappedPytorchModel(model)
        except (NameError, AttributeError):
            module_logger.debug(
                'Could not import torch, required if using a pytorch model')
        if _is_fastai_tabular_model(model):
            model = WrappedFastAITabularModel(model)
        if is_sequential(model):
            model = WrappedTensorflowModel(model)
        if _classifier_without_proba(model):
            model = WrappedClassificationWithoutProbaModel(model)
        eval_function, eval_ml_domain = _eval_model(
            model, examples, model_task)
        if eval_ml_domain == ModelTask.CLASSIFICATION:
            return WrappedClassificationModel(model, eval_function, examples), \
                eval_ml_domain
        else:
            return WrappedRegressionModel(model, eval_function, examples), \
                eval_ml_domain


def _classifier_without_proba(model):
    """Returns True if the given model is a classifier without predict_proba,
        eg SGDClassifier.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function
    :return: True if the given model is a classifier without predict_proba.
    :rtype: bool
    """
    return isinstance(model, SGDClassifier) and not \
        hasattr(model, SKLearn.PREDICT_PROBA)
