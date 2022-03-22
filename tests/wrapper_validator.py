# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utilities for validating wrapped models."""

from ml_wrappers.common.constants import SKLearn


def validate_wrapped_classification_model(wrapped_model, X_test):
    # validate wrapped model has predict and predict_proba functions
    assert hasattr(wrapped_model, SKLearn.PREDICT)
    assert hasattr(wrapped_model, SKLearn.PREDICT_PROBA)
    # validate we can call the model on the dataset
    predictions = wrapped_model.predict(X_test)
    probabilities = wrapped_model.predict_proba(X_test)
    # validate predictions and probabilities have correct shape
    assert len(predictions.shape) == 1
    assert len(probabilities.shape) == 2


def validate_wrapped_regression_model(wrapped_model, X_test):
    # validate wrapped model has predict function and NO predict_proba function
    assert hasattr(wrapped_model, SKLearn.PREDICT)
    assert not hasattr(wrapped_model, SKLearn.PREDICT_PROBA)
    # validate we can call the model on the dataset
    predictions = wrapped_model.predict(X_test)
    # validate predictions have correct shape
    assert len(predictions.shape) == 1
