# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os
import tempfile

import pytest
from common_utils import (create_complex_titanic_data, create_housing_data,
                          create_iris_data, create_simple_titanic_data)
from constants import DatasetConstants

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)


@pytest.fixture()
def _clean_dir():
    new_path = tempfile.mkdtemp()
    print("tmp test directory: " + new_path)
    os.chdir(new_path)


@pytest.fixture(scope='session')
def iris():
    x_train, x_test, y_train, y_test, features, classes = create_iris_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features,
        DatasetConstants.CLASSES: classes
    }


@pytest.fixture(scope='session')
def housing():
    x_train, x_test, y_train, y_test, features = create_housing_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features
    }


@pytest.fixture(scope='session')
def titanic_simple():
    x_train, x_test, y_train, y_test, numeric, categorical = create_simple_titanic_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.NUMERIC: numeric,
        DatasetConstants.CATEGORICAL: categorical
    }


@pytest.fixture(scope='session')
def titanic_complex():
    x_train, x_test, y_train, y_test = create_complex_titanic_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test
    }
