# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines common model utilities."""

from ml_wrappers.common.warnings_suppressor import shap_warnings_suppressor

with shap_warnings_suppressor():
    try:
        from shap.utils import safe_isinstance
        shap_installed = True
    except BaseException:  # noqa: B036
        shap_installed = False


MULTILABEL_THRESHOLD = 0.5


def _is_transformers_pipeline(model):
    """Checks if the model is a transformers pipeline.

    :param model: The model to check.
    :type model: object
    :return: True if the model is a transformers pipeline, False otherwise.
    :rtype: bool
    """
    return shap_installed and safe_isinstance(
        model, "transformers.pipelines.Pipeline")


def _is_callable_pipeline(model):
    """Checks if the model is a callable pipeline.

    Returns false if the model has a predict and predict_proba method.

    :param model: The model to check.
    :type model: object
    :return: True if the model is a callable pipeline, False otherwise.
    :rtype: bool
    """
    has_predict = hasattr(model, 'predict')
    has_predict_proba = hasattr(model, 'predict_proba')
    return callable(model) and not has_predict and not has_predict_proba
