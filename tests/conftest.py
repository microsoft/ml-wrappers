# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os
import tempfile

import pytest

try:
    from common_utils import create_cancer_data_booleans
except ModuleNotFoundError:
    print("Could not import common_utils, may be running minimal tests")
    pass

from constants import DatasetConstants
from rai_test_utils.datasets.tabular import (
    create_cancer_data, create_complex_titanic_data, create_diabetes_data,
    create_energy_data, create_housing_data, create_iris_data,
    create_multiclass_classification_dataset, create_simple_titanic_data,
    create_wine_data)

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
        DatasetConstants.X_TRAIN: x_train.values,
        DatasetConstants.X_TEST: x_test.values,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features,
        DatasetConstants.CLASSES: classes
    }


@pytest.fixture(scope='session')
def cancer():
    x_train, x_test, y_train, y_test, features, classes = create_cancer_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features,
        DatasetConstants.CLASSES: classes
    }


@pytest.fixture(scope='session')
def cancer_booleans():
    x_train, x_test, y_train, y_test, features, classes = create_cancer_data_booleans()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features,
        DatasetConstants.CLASSES: classes
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


@pytest.fixture(scope='session')
def wine():
    x_train, x_test, y_train, y_test, features, classes = create_wine_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features,
        DatasetConstants.CLASSES: classes
    }


@pytest.fixture(scope='session')
def multiclass_classification():
    x_train, y_train, x_test, y_test, classes = \
        create_multiclass_classification_dataset()
    feature_names = ["col" + str(i) for i in list(range(x_train.shape[1]))]

    return {
        DatasetConstants.X_TRAIN: x_train.values,
        DatasetConstants.X_TEST: x_test.values,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: feature_names,
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
def energy():
    x_train, x_test, y_train, y_test, features = create_energy_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features
    }


@pytest.fixture(scope='session')
def diabetes():
    x_train, x_test, y_train, y_test, features = create_diabetes_data()
    return {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features
    }
