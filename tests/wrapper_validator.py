# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utilities for validating wrapped models."""

from ml_wrappers.common.constants import ModelTask, SKLearn
from ml_wrappers.model import WrappedPytorchModel, WrappedTensorflowModel

PREDICT_CLASSES = 'predict_classes'


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


def validate_wrapped_question_answering_model(wrapped_model, X_test):
    # validate wrapped model has predict and predict_proba functions
    assert hasattr(wrapped_model, SKLearn.PREDICT)
    assert not hasattr(wrapped_model, SKLearn.PREDICT_PROBA)
    # validate we can call the model on the dataset
    predictions = wrapped_model.predict(X_test)
    # validate predictions have correct shape
    assert len(predictions) == len(X_test)
    assert isinstance(predictions[0], str)


def validate_wrapped_tf_model(wrapped_tf_model, X_test, model_task):
    assert isinstance(wrapped_tf_model, WrappedTensorflowModel)
    validate_wrapped_pred_classes_model(wrapped_tf_model, X_test, model_task)


def validate_wrapped_pytorch_model(wrapped_pytorch_model, X_test, model_task):
    assert isinstance(wrapped_pytorch_model, WrappedPytorchModel)
    validate_wrapped_pred_classes_model(wrapped_pytorch_model, X_test, model_task)


def validate_wrapped_pred_classes_model(wrapped_model, X_test, model_task):
    assert hasattr(wrapped_model, SKLearn.PREDICT)
    assert hasattr(wrapped_model, SKLearn.PREDICT_PROBA)
    assert hasattr(wrapped_model, PREDICT_CLASSES)
    # validate we can call the model on the dataset
    if model_task == ModelTask.CLASSIFICATION:
        probabilities = wrapped_model.predict_proba(X_test)
        predictions = wrapped_model.predict_classes(X_test)
        # validate predictions and probabilities have correct shape
        assert len(predictions.shape) == 1
        assert len(probabilities.shape) == 2
    else:
        predictions = wrapped_model.predict(X_test)
        # validate predictions have correct shape
        print(predictions.shape)
        assert len(predictions.shape) == 1 or predictions.shape[1] == 1
