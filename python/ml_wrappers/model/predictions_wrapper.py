# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines classes to wrap the training/test data and the corresponding predictions from the model."""

from typing import Optional

import numpy as np
import pandas as pd

TARGET = 'target'


class PredictionsModelWrapper:
    """Model wrapper to wrap the samples used to train the models
    and the predictions of the model. This wrapper is useful when
    it is not possible to load the model.
    """
    def __init__(self, test_data: pd.DataFrame, y_pred: np.ndarray):
        """Creates a PredictionsModelWrapper object.

        :param test_data: The data that was used to train the model.
        :type test_data: pd.DataFrame
        :param y_pred: Predictions of the model.
        :type y_pred: np.ndarray
        """
        if not isinstance(test_data, pd.DataFrame):
            raise DataValidationException(
                "Expecting a pandas dataframe for test_data")
        if not isinstance(y_pred, np.ndarray):
            raise DataValidationException(
                "Expecting a numpy array for y_pred")
        if len(test_data) != len(y_pred):
            raise DataValidationException(
                "The number of instances in test data "
                "do not match with number of predictions")
        self._combined_data = test_data.copy()
        self._combined_data[TARGET] = y_pred

    def _get_filtered_data(
            self, query_test_data_row: pd.DataFrame) -> pd.DataFrame:
        """Return the filtered data based on the query data.

        :param query_test_data_row: The query data instance.
        :type query_test_data_row: pd.DataFrame
        :return: The filtered dataframe based on the values in query data.
        :rtype: pd.DataFrame
        """
        queries = []
        for column_name, column_data in query_test_data_row.squeeze().iteritems():
            if isinstance(column_data, str):
                queries.append("`{}` == '{}'".format(
                    column_name, column_data))
            else:
                queries.append("`{}` == {}".format(
                    column_name, column_data))

        queries_str = '(' + ') & ('.join(queries) + ')'
        filtered_df = self._combined_data.query(queries_str)

        if len(filtered_df) == 0:
            raise EmptyDataException(
                "The query data was not found in the combined dataset")

        return filtered_df

    def predict(self, query_test_data: pd.DataFrame) -> np.ndarray:
        """Return the predictions based on the query data.

        :param query_test_data: The data for which the predictions need to
            be returned.
        :type query_test_data: pd.DataFrame
        :return: Predictions of the model.
        :rtype: np.ndarray
        """
        if not isinstance(query_test_data, pd.DataFrame):
            raise DataValidationException(
                "Expecting a pandas dataframe for query_test_data")
        prediction_output = []
        for index in range(0, len(query_test_data)):
            filtered_df = self._get_filtered_data(
                query_test_data[index:index + 1])
            prediction_output.append(filtered_df[TARGET].values[0])

        return np.array(prediction_output)

    def __setstate__(self, state):
        """Set the state of the class object so that the wrapped
        model is serializable via pickle.

        :param state: A dictionary of deserialized state.
        :type state: Dict
        """
        self._combined_data = state["_combined_data"]

    def __getstate__(self):
        """Return the state so that the wrapped model is
        serializable via pickle.

        :return: The state to be pickled.
        :rtype: Dict
        """
        state = {}
        state["_combined_data"] = self._combined_data
        return state


class PredictionsModelWrapperRegression(PredictionsModelWrapper):
    """Model wrapper to wrap the samples used to train the models
    and the predictions of the model for regression tasks.
    """
    def __init__(self, test_data: pd.DataFrame, y_pred: np.ndarray):
        """Creates a PredictionsModelWrapperRegression object.

        :param test_data: The data that was used to train the model.
        :type test_data: pd.DataFrame
        :param y_pred: Predictions of the model.
        :type y_pred: np.ndarray
        """
        super(PredictionsModelWrapperRegression, self).__init__(
            test_data, y_pred)


class PredictionsModelWrapperClassification(PredictionsModelWrapper):
    """Model wrapper to wrap the samples used to train the models
    and the predictions of the model for classification tasks.
    """
    def __init__(self, test_data: pd.DataFrame, y_pred: np.ndarray,
                 y_pred_proba: Optional[np.ndarray] = None):
        """Creates a PredictionsModelWrapperClassification object.

        :param test_data: The data that was used to train the model.
        :type test_data: pd.DataFrame
        :param y_pred: Predictions of the model.
        :type y_pred: np.ndarray
        :param y_pred_proba: Prediction probabilities of the model.
        :type y_pred_proba: np.ndarray
        """
        super(PredictionsModelWrapperClassification, self).__init__(
            test_data, y_pred)
        self._num_classes = None
        if y_pred_proba is not None:
            if not isinstance(y_pred_proba, np.ndarray):
                raise DataValidationException(
                    "Expecting a numpy array for y_pred_proba")
            for i in range(0, len(y_pred_proba[0])):
                self._combined_data[
                    TARGET + '_' + str(i)] = y_pred_proba[:, i]
            self._num_classes = len(y_pred_proba[0])

    def predict_proba(self, query_test_data: pd.DataFrame) -> np.ndarray:
        """Return the prediction probabilities based on the query data.

        :param query_test_data: The data for which the prediction
            probabilities need to be returned.
        :type query_test_data: pd.DataFrame
        :return: Prediction probabilities of the model.
        :rtype: np.ndarray
        """
        if not isinstance(query_test_data, pd.DataFrame):
            raise DataValidationException(
                "Expecting a pandas dataframe for query_test_data")
        if self._num_classes is None:
            raise DataValidationException(
                "Model wrapper configured without prediction probabilities"
            )
        prediction_output = []
        for index in range(0, len(query_test_data)):
            filtered_df = self._get_filtered_data(
                query_test_data[index:index + 1])
            classes_output = []
            for i in range(self._num_classes):
                classes_output.append(
                    filtered_df[TARGET + '_' + str(i)].values[0])
            prediction_output.append(classes_output)

        return np.array(prediction_output)

    def __setstate__(self, state):
        """Set the state of the class object so that the wrapped
        model is serializable via pickle.

        :param state: A dictionary of deserialized state.
        :type state: Dict
        """
        super().__setstate__(state)
        self._num_classes = state["_num_classes"]

    def __getstate__(self):
        """Return the state so that the wrapped model is
        serializable via pickle.

        :return: The state to be pickled.
        :rtype: Dict
        """
        state = super().__getstate__()
        state["_num_classes"] = self._num_classes
        return state


class EmptyDataException(Exception):
    """An exception indicating that some operation produced empty data.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = 'Empty data exception'


class DataValidationException(Exception):
    """An exception indicating that some user supplied data is not valid.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = 'Invalid data'
