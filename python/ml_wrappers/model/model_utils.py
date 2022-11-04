# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines common model utilities."""

from ml_wrappers.common.warnings_suppressor import shap_warnings_suppressor

with shap_warnings_suppressor():
    try:
        from shap.utils import safe_isinstance
        shap_installed = True
    except BaseException:
        shap_installed = False


MULTILABEL_THRESHOLD = 0.5


def _is_transformers_pipeline(model):
    return shap_installed and safe_isinstance(
        model, "transformers.pipelines.Pipeline")
