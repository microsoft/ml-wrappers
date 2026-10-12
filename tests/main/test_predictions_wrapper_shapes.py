# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for prediction shape normalization."""

import pickle

import numpy as np
import pandas as pd
import pytest
from ml_wrappers.model.predictions_wrapper import (
    DataValidationException, PredictionsModelWrapperClassification,
    PredictionsModelWrapperRegression)


@pytest.mark.parametrize('should_construct_pandas_query', [True, False])
@pytest.mark.parametrize(
    'model_predict',
    [
        np.array([[False], [True], [False]]),
        np.array([['class-a'], ['class-b'], ['class-c']]),
        np.array([['class-a'], ['class-b'], ['class-c']], dtype=object),
    ],
    ids=['boolean', 'string', 'object'],
)
def test_classification_normalizes_single_column_predictions(
        model_predict, should_construct_pandas_query):
    test_data = pd.DataFrame({'feature': [1, 2, 3]})
    expected = model_predict[:, 0]

    model_wrapper = PredictionsModelWrapperClassification(
        test_data, model_predict,
        should_construct_pandas_query=should_construct_pandas_query)

    result = model_wrapper.predict(test_data)
    assert result.shape == (len(test_data),)
    np.testing.assert_array_equal(result, expected)
    assert all(not isinstance(value, np.ndarray) for value in result)
    assert all(hash(value) is not None for value in result)

    deserialized_wrapper = pickle.loads(pickle.dumps(model_wrapper))
    np.testing.assert_array_equal(
        deserialized_wrapper.predict(test_data),
        expected,
    )


@pytest.mark.parametrize('should_construct_pandas_query', [True, False])
def test_regression_normalizes_single_column_predictions(
        should_construct_pandas_query):
    test_data = pd.DataFrame({'feature': [1, 2, 3]})
    model_predict = np.array([[1.5], [2.5], [3.5]])
    expected = model_predict[:, 0]

    model_wrapper = PredictionsModelWrapperRegression(
        test_data, model_predict,
        should_construct_pandas_query=should_construct_pandas_query)

    result = model_wrapper.predict(test_data)
    assert result.shape == (len(test_data),)
    np.testing.assert_array_equal(result, expected)

    deserialized_wrapper = pickle.loads(pickle.dumps(model_wrapper))
    np.testing.assert_array_equal(
        deserialized_wrapper.predict(test_data),
        expected,
    )


@pytest.mark.parametrize(
    'wrapper_type',
    [
        PredictionsModelWrapperClassification,
        PredictionsModelWrapperRegression,
    ],
)
def test_rejects_multiple_prediction_columns(wrapper_type):
    test_data = pd.DataFrame({'feature': [1, 2, 3]})
    model_predict = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.4, 0.6],
    ])

    with pytest.raises(
            DataValidationException,
            match="Expecting y_pred to be one-dimensional or a "
                  "two-dimensional array with one column"):
        wrapper_type(test_data, model_predict)
