# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for ml-wrappers compatibility with supported NumPy versions."""

import numpy as np
import pandas as pd
from ml_wrappers.dataset import DatasetWrapper
from ml_wrappers.model.predictions_wrapper import \
    PredictionsModelWrapperClassification


def test_dataset_and_predictions_support_numpy_1_and_2():
    """Verify dataset and prediction wrappers with NumPy 1.x and 2.x."""
    assert int(np.__version__.split('.')[0]) in (1, 2)

    data = pd.DataFrame({
        'numeric': [1.0, np.nan, 3.0],
        'category': ['a', 'b', 'c']
    })
    predictions = np.array([0, 1, 0])
    probabilities = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3]
    ])

    dataset_wrapper = DatasetWrapper(data)
    model_wrapper = PredictionsModelWrapperClassification(
        data, predictions, probabilities)

    assert dataset_wrapper.typed_dataset.equals(data)
    assert dataset_wrapper.string_index() is not None
    np.testing.assert_array_equal(model_wrapper.predict(data), predictions)
    np.testing.assert_allclose(
        model_wrapper.predict_proba(data), probabilities)
