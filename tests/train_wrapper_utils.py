# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utilities for calling the wrap_model function and validating the results."""

import pandas as pd
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.dataset.dataset_wrapper import DatasetWrapper

from constants import DatasetConstants
from wrapper_validator import (validate_wrapped_classification_model,
                               validate_wrapped_regression_model)


def train_classification_model_numpy(model_initializer, dataset,
                                     use_dataset_wrapper=True):
    X_train = dataset[DatasetConstants.X_TRAIN]
    X_test = dataset[DatasetConstants.X_TEST]
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped,
                               model_task=ModelTask.CLASSIFICATION)
    validate_wrapped_classification_model(wrapped_model, X_test)


def train_classification_model_pandas(model_initializer, dataset,
                                      use_dataset_wrapper=True,
                                      validate_single_row=False):
    X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                           columns=dataset[DatasetConstants.FEATURES])
    X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                          columns=dataset[DatasetConstants.FEATURES])
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped,
                               model_task=ModelTask.CLASSIFICATION)
    if validate_single_row:
        validate_wrapped_classification_model(wrapped_model, X_test.iloc[0:1])
    validate_wrapped_classification_model(wrapped_model, X_test)


def train_regression_model_numpy(model_initializer, dataset,
                                 use_dataset_wrapper=True):
    X_train = dataset[DatasetConstants.X_TRAIN]
    X_test = dataset[DatasetConstants.X_TEST]
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped,
                               model_task=ModelTask.REGRESSION)
    validate_wrapped_regression_model(wrapped_model, X_test)


def train_regression_model_pandas(model_initializer, dataset,
                                  use_dataset_wrapper=True):
    X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                           columns=dataset[DatasetConstants.FEATURES])
    X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                          columns=dataset[DatasetConstants.FEATURES])
    y_train = dataset[DatasetConstants.Y_TRAIN]
    model = model_initializer(X_train, y_train)
    if use_dataset_wrapper:
        X_test_wrapped = DatasetWrapper(X_test)
    else:
        X_test_wrapped = X_test
    wrapped_model = wrap_model(model, X_test_wrapped,
                               model_task=ModelTask.REGRESSION)
    validate_wrapped_regression_model(wrapped_model, X_test)
