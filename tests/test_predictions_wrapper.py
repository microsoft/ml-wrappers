# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for predictions model wrapper."""

import numpy as np
import pandas as pd
from common_utils import create_lightgbm_classifier, create_lightgbm_regressor
from constants import DatasetConstants
from ml_wrappers.model.predictions_wrapper import (
    ModelWrapperPredictionsClassification, ModelWrapperPredictionsRegression)


class TestPredictionsWrapper:

    def verify_predict_outputs(self, model, model_wrapper, test_data):
        model_predict_output = model.predict(test_data)
        model_wrapper_predict_output = model_wrapper.predict(test_data)
        np.all(model_predict_output == model_wrapper_predict_output)

    def verify_predict_proba_outputs(self, model, model_wrapper, test_data):
        model_predict_proba_output = model.predict_proba(test_data)
        model_wrapper_predict_proba_output = model_wrapper.predict_proba(test_data)
        np.all(model_predict_proba_output == model_wrapper_predict_proba_output)

    def test_prediction_wrapper_classification(self, iris):
        dataset = iris
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_classifier(X_train, y_train)

        model_predict = model.predict(X_test)
        model_predict_proba = model.predict_proba(X_test)

        model_wrapper = ModelWrapperPredictionsClassification(
            X_test, model_predict, model_predict_proba)

        self.verify_predict_outputs(model, model_wrapper, X_test)
        self.verify_predict_proba_outputs(model, model_wrapper, X_test)

    def test_prediction_wrapper_regression(self, housing):
        dataset = housing
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_regressor(X_train, y_train)

        model_predict = model.predict(X_test)

        model_wrapper = ModelWrapperPredictionsRegression(
            X_test, model_predict)

        self.verify_predict_outputs(model, model_wrapper, X_test)
