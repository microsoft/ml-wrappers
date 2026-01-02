# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for predictions model wrapper."""

import pickle

import numpy as np
import pandas as pd
import pytest
from common_utils import (create_lightgbm_classifier,
                          create_lightgbm_regressor, create_titanic_pipeline)
from constants import DatasetConstants
from ml_wrappers.dataset import DatasetWrapper
from ml_wrappers.model.predictions_wrapper import (
    DataValidationException, EmptyDataException,
    PredictionsModelWrapperClassification, PredictionsModelWrapperRegression)


class TestPredictionsWrapper:

    def verify_predict_outputs(self, model, model_wrapper, test_data):
        model_predict_output = model.predict(test_data)
        model_wrapper_predict_output = model_wrapper.predict(test_data)
        np.all(model_predict_output == model_wrapper_predict_output)

    def verify_predict_proba_outputs(self, model, model_wrapper, test_data):
        model_predict_proba_output = model.predict_proba(test_data)
        model_wrapper_predict_proba_output = model_wrapper.predict_proba(test_data)
        np.all(model_predict_proba_output == model_wrapper_predict_proba_output)

    def verify_pickle_serialization(self, model, model_wrapper, test_data):
        new_model_wrapper = pickle.loads(pickle.dumps(model_wrapper))
        self.verify_predict_outputs(model, new_model_wrapper, test_data)
        if hasattr(new_model_wrapper, "predict_proba"):
            self.verify_predict_proba_outputs(model, new_model_wrapper, test_data)


@pytest.mark.parametrize('should_construct_pandas_query', [True, False])
class TestPredictionsWrapperClassification(TestPredictionsWrapper):
    @pytest.mark.parametrize('dataset_name', ['iris', 'titanic', 'cancer', 'wine', 'multiclass'])
    def test_prediction_wrapper_classification(
            self, iris, titanic_simple, cancer, cancer_booleans, wine,
            multiclass_classification, dataset_name, should_construct_pandas_query):
        dataset_to_fixture_dict = {
            'iris': iris,
            'titanic': titanic_simple,
            'cancer': cancer,
            'cancer_booleans': cancer_booleans,
            'multiclass': multiclass_classification,
            'wine': wine
        }
        dataset = dataset_to_fixture_dict[dataset_name]

        if dataset_name != 'titanic':
            features = dataset[DatasetConstants.FEATURES]
        else:
            features = dataset[DatasetConstants.NUMERIC] + dataset[DatasetConstants.CATEGORICAL]

        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=features)
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=features)

        y_train = dataset[DatasetConstants.Y_TRAIN]
        if dataset_name != 'titanic':
            model = create_lightgbm_classifier(X_train, y_train)
        else:
            model = create_titanic_pipeline(X_train, y_train)

        model_predict = model.predict(X_test)
        model_predict_proba = model.predict_proba(X_test)

        model_wrapper = PredictionsModelWrapperClassification(
            X_test, model_predict, model_predict_proba,
            should_construct_pandas_query=should_construct_pandas_query)

        self.verify_predict_outputs(model, model_wrapper, X_test)
        self.verify_predict_proba_outputs(model, model_wrapper, X_test)
        self.verify_pickle_serialization(model, model_wrapper, X_test)

    def test_prediction_wrapper_unsupported_scenarios(
            self, iris, should_construct_pandas_query):
        dataset = iris
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_classifier(X_train, y_train)

        model_predict = model.predict(X_test)
        model_predict_proba = model.predict_proba(X_test)

        with pytest.raises(
                DataValidationException,
                match="Expecting a pandas dataframe for test_data"):
            PredictionsModelWrapperClassification(
                X_test.values, model_predict, model_predict_proba,
                should_construct_pandas_query=should_construct_pandas_query
            )

        with pytest.raises(
                DataValidationException,
                match="Expecting a numpy array for y_pred"):
            PredictionsModelWrapperClassification(
                X_test, model_predict.tolist(), model_predict_proba,
                should_construct_pandas_query=should_construct_pandas_query
            )

        with pytest.raises(
                DataValidationException,
                match="Expecting a numpy array for y_pred_proba"):
            PredictionsModelWrapperClassification(
                X_test, model_predict, model_predict_proba.tolist(),
                should_construct_pandas_query=should_construct_pandas_query
            )

        with pytest.raises(
                DataValidationException,
                match="The number of instances in test data "
                      "do not match with number of predictions"):
            PredictionsModelWrapperClassification(
                X_test.iloc[0:len(X_test) - 1], model_predict, model_predict_proba,
                should_construct_pandas_query=should_construct_pandas_query
            )

        with pytest.raises(
                DataValidationException,
                match="The number of instances in test data "
                      "do not match with number of prediction probabilities"):
            PredictionsModelWrapperClassification(
                X_test.iloc[0:len(X_test) - 1], model_predict[0:len(X_test) - 1], model_predict_proba,
                should_construct_pandas_query=should_construct_pandas_query
            )

    def test_prediction_wrapper_unsupported_predict_scenarios(self, iris, should_construct_pandas_query):
        dataset = iris
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_classifier(X_train, y_train)

        model_predict = model.predict(X_test)
        model_predict_proba = model.predict_proba(X_test)

        model_wrapper = PredictionsModelWrapperClassification(
            X_test, model_predict, model_predict_proba,
            should_construct_pandas_query=should_construct_pandas_query)
        with pytest.raises(
                DataValidationException,
                match="Expecting a pandas dataframe for query_test_data"):
            model_wrapper.predict(X_test.values)
        with pytest.raises(
                DataValidationException,
                match="Expecting a pandas dataframe for query_test_data"):
            model_wrapper.predict_proba(X_test.values)

        model_wrapper_without_predict_proba = \
            PredictionsModelWrapperClassification(X_test, model_predict)
        with pytest.raises(
                DataValidationException,
                match="Model wrapper configured without prediction probabilities"):
            model_wrapper_without_predict_proba.predict_proba(X_test)

    def test_prediction_wrapper_unseen_data_predict_scenarios(
            self, iris, should_construct_pandas_query):
        dataset = iris
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_classifier(X_train, y_train)

        model_predict = model.predict(X_test[0:len(X_test) - 1])
        model_predict_proba = model.predict_proba(X_test[0:len(X_test) - 1])

        model_wrapper = PredictionsModelWrapperClassification(
            X_test[0:len(X_test) - 1], model_predict, model_predict_proba,
            should_construct_pandas_query=should_construct_pandas_query)

        with pytest.raises(
                EmptyDataException,
                match="The query data was not found in the combined dataset"):
            model_wrapper.predict(X_test[len(X_test) - 1:len(X_test)])

        with pytest.raises(
                EmptyDataException,
                match="The query data was not found in the combined dataset"):
            model_wrapper.predict_proba(X_test[len(X_test) - 1:len(X_test)])


@pytest.mark.parametrize('should_construct_pandas_query', [True, False])
class TestPredictionsWrapperRegression(TestPredictionsWrapper):
    @pytest.mark.parametrize('dataset_name', ['housing', 'energy', 'diabetes'])
    def test_prediction_wrapper_regression(
            self, should_construct_pandas_query,
            dataset_name, housing, energy, diabetes):
        dataset_to_fixture_dict = {
            'housing': housing,
            'energy': energy,
            'diabetes': diabetes
        }
        dataset = dataset_to_fixture_dict[dataset_name]

        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])

        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_regressor(X_train, y_train)

        model_predict = model.predict(X_test)

        model_wrapper = PredictionsModelWrapperRegression(
            X_test, model_predict,
            should_construct_pandas_query=should_construct_pandas_query)

        self.verify_predict_outputs(model, model_wrapper, X_test)
        assert not hasattr(model_wrapper, "predict_proba")
        self.verify_pickle_serialization(model, model_wrapper, X_test)

    @pytest.mark.parametrize('length_test_set', [10, 100, 200, 300, 400, 500, 1000])
    def test_prediction_wrapper_regression_perf(self, housing, length_test_set, should_construct_pandas_query):
        dataset = housing

        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])

        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_regressor(X_train, y_train)

        model_predict = model.predict(X_test[0:length_test_set])

        model_wrapper = PredictionsModelWrapperRegression(
            X_test[0:length_test_set], model_predict, should_construct_pandas_query=should_construct_pandas_query)

        self.verify_predict_outputs(model, model_wrapper, X_test[0:length_test_set])
        assert not hasattr(model_wrapper, "predict_proba")
        self.verify_pickle_serialization(model, model_wrapper, X_test[0:length_test_set])

    def test_data_with_missing_values(self, should_construct_pandas_query):
        data = {
            'Age': [25, 30, 22, 28, np.nan],
            'Income': [50000, 60000, 40000, np.nan, 75000],
            'Education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master'],
            'Employment': ['Full-time', 'Part-time', 'Unemployed', 'Full-time', 'Part-time'],
            'Has_Car': [True, False, True, True, False],
            'Credit_Score': [750, 800, np.nan, 720, 690],
            'Loan_Approved': [1, 1, 0, 0, 1]  # 1 = Approved, 0 = Not Approved
        }

        df = pd.DataFrame(data)
        X_train = df.drop('Loan_Approved', axis=1)
        y_train = df['Loan_Approved'].values

        model_wrapper = PredictionsModelWrapperRegression(
            X_train, y_train, should_construct_pandas_query=should_construct_pandas_query)
        self.verify_predict_outputs(model_wrapper, model_wrapper, X_train)
        assert not hasattr(model_wrapper, "predict_proba")
        self.verify_pickle_serialization(model_wrapper, model_wrapper, X_train)

    def test_prediction_wrapper_with_reset_index_query(
            self, housing, should_construct_pandas_query):
        """Test that query data with reset index still matches original data.

        This tests the fix for the issue where DatasetWrapper converts data
        to numpy and back, resetting the index. The wrapper should match
        by values, not by index.
        """
        dataset = housing

        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])

        # Give X_test a non-sequential index to trigger the bug
        X_test.index = range(300, 300 + len(X_test))

        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_regressor(X_train, y_train)

        model_predict = model.predict(X_test)

        model_wrapper = PredictionsModelWrapperRegression(
            X_test, model_predict,
            should_construct_pandas_query=should_construct_pandas_query)

        # Simulate what DatasetWrapper does: convert to numpy and back
        # This resets the index to 0, 1, 2, ... instead of original index [300, 301, ...]
        dataset_wrapper = DatasetWrapper(X_test)
        wrapped_data = dataset_wrapper.typed_wrapper_func(dataset_wrapper.dataset)

        # The wrapped data should have reset index
        assert list(wrapped_data.index) == list(range(len(X_test)))
        # But original X_test has different index [300, 301, ...]
        # Query should still work despite index mismatch
        self.verify_predict_outputs(model, model_wrapper, wrapped_data)


@pytest.mark.parametrize('should_construct_pandas_query', [True, False])
class TestPredictionsWrapperIndexMismatch(TestPredictionsWrapper):
    """Tests specifically for index mismatch scenarios."""

    def test_classification_with_non_sequential_index(
            self, iris, should_construct_pandas_query):
        """Test classification wrapper with non-sequential DataFrame index."""
        dataset = iris
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_classifier(X_train, y_train)

        # Give X_test a non-sequential index (like real data after train/test split)
        X_test.index = range(100, 100 + len(X_test))

        # Create wrapper with non-sequential index
        model_predict = model.predict(X_test)
        model_predict_proba = model.predict_proba(X_test)

        model_wrapper = PredictionsModelWrapperClassification(
            X_test, model_predict, model_predict_proba,
            should_construct_pandas_query=should_construct_pandas_query)

        # Query with reset index (simulates DatasetWrapper transformation)
        # This has index [0,1,2,3,4] while wrapper has [100,101,102,103,104]
        query_data = X_test.iloc[0:5].reset_index(drop=True)

        # Should still work - matching by values, not index
        result = model_wrapper.predict(query_data)
        expected = model.predict(X_test.iloc[0:5])
        np.testing.assert_array_equal(result, expected)

        result_proba = model_wrapper.predict_proba(query_data)
        expected_proba = model.predict_proba(X_test.iloc[0:5])
        np.testing.assert_array_almost_equal(result_proba, expected_proba)

    def test_regression_with_non_sequential_index(
            self, housing, should_construct_pandas_query):
        """Test regression wrapper with non-sequential DataFrame index."""
        dataset = housing
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_regressor(X_train, y_train)

        # Give X_test a non-sequential index
        X_test.index = range(500, 500 + len(X_test))

        model_predict = model.predict(X_test)

        model_wrapper = PredictionsModelWrapperRegression(
            X_test, model_predict,
            should_construct_pandas_query=should_construct_pandas_query)

        # Query with reset index (index [0,1,2,3,4] vs wrapper's [500,501,...])
        query_data = X_test.iloc[0:5].reset_index(drop=True)

        result = model_wrapper.predict(query_data)
        expected = model.predict(X_test.iloc[0:5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_dataset_wrapper_transformation(
            self, iris, should_construct_pandas_query):
        """Test the exact scenario that caused the bug: DatasetWrapper transformation."""
        dataset = iris
        X_train = pd.DataFrame(data=dataset[DatasetConstants.X_TRAIN],
                               columns=dataset[DatasetConstants.FEATURES])
        X_test = pd.DataFrame(data=dataset[DatasetConstants.X_TEST],
                              columns=dataset[DatasetConstants.FEATURES])
        y_train = dataset[DatasetConstants.Y_TRAIN]
        model = create_lightgbm_classifier(X_train, y_train)

        # Give X_test a non-sequential index (like after concat of test+train)
        X_test.index = range(200, 200 + len(X_test))

        model_predict = model.predict(X_test)
        model_predict_proba = model.predict_proba(X_test)

        model_wrapper = PredictionsModelWrapperClassification(
            X_test, model_predict, model_predict_proba,
            should_construct_pandas_query=should_construct_pandas_query)

        # Simulate DatasetWrapper transformation (numpy -> DataFrame with reset index)
        dataset_wrapper = DatasetWrapper(X_test)
        internal_data = dataset_wrapper.dataset  # numpy array
        wrapped_query = dataset_wrapper.typed_wrapper_func(internal_data[0:1])

        # This is the exact scenario that was failing before the fix
        result = model_wrapper.predict_proba(wrapped_query)
        expected = model.predict_proba(X_test.iloc[0:1])
        np.testing.assert_array_almost_equal(result, expected)
