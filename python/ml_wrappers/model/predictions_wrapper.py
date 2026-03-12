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
    def __init__(self, test_data: pd.DataFrame, y_pred: np.ndarray,
                 should_construct_pandas_query: Optional[bool] = True):
        """Creates a PredictionsModelWrapper object.

        :param test_data: The data that was used to train the model.
        :type test_data: pd.DataFrame
        :param y_pred: Predictions of the model.
        :type y_pred: np.ndarray
        :param should_construct_pandas_query: Whether to use pandas query()
            function to filter the data. If set to False, the data will
            be filtered using iloc(). Defaults to True.
        :type should_construct_pandas_query: Optional[bool]
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
        self._feature_names = list(test_data.columns)
        self._combined_data = test_data.copy()
        self._combined_data[TARGET] = y_pred
        self._should_construct_pandas_query = should_construct_pandas_query
        # Build hash-based index for O(1) lookups on large datasets
        self._row_hash_to_idx = self._build_row_hash_index()

    def _compute_row_hash(self, row_values) -> int:
        """Compute a hash for a row of values.

        Handles NaN values and various data types consistently.

        :param row_values: The row values to hash (tuple or array-like).
        :type row_values: tuple or array-like
        :return: Hash value for the row.
        :rtype: int
        """
        # Convert to tuple, handling NaN values specially
        hashable_values = []
        for val in row_values:
            if pd.isna(val):
                hashable_values.append(('__NA__', type(val).__name__))
            elif isinstance(val, (np.floating, float)):
                # Round floats to avoid floating point precision issues
                hashable_values.append(round(float(val), 10))
            elif isinstance(val, np.ndarray):
                hashable_values.append(tuple(val.flatten()))
            else:
                try:
                    hashable_values.append(val)
                except TypeError:
                    # Fallback for unhashable types
                    hashable_values.append(str(val))
        return hash(tuple(hashable_values))

    def _build_row_hash_index(self) -> dict:
        """Build a hash-based index mapping row hashes to DataFrame indices.

        This enables O(1) lookups instead of O(n) filtering for each query row.

        :return: Dictionary mapping row hash to list of matching indices.
        :rtype: dict
        """
        hash_to_idx = {}
        feature_data = self._combined_data[self._feature_names]
        for idx in range(len(feature_data)):
            row_values = tuple(feature_data.iloc[idx].values)
            row_hash = self._compute_row_hash(row_values)
            if row_hash not in hash_to_idx:
                hash_to_idx[row_hash] = []
            hash_to_idx[row_hash].append(idx)
        return hash_to_idx

    def _lookup_by_hash(self, query_row: pd.DataFrame) -> Optional[int]:
        """Look up a row using the hash index.

        :param query_row: Single row DataFrame to look up.
        :type query_row: pd.DataFrame
        :return: Index in combined_data if found, None otherwise.
        :rtype: Optional[int]
        """
        row_values = tuple(query_row[self._feature_names].iloc[0].values)
        row_hash = self._compute_row_hash(row_values)

        if row_hash not in self._row_hash_to_idx:
            return None

        # Hash collision handling: verify actual match
        candidate_indices = self._row_hash_to_idx[row_hash]
        query_values = query_row[self._feature_names].iloc[0]

        for idx in candidate_indices:
            stored_values = self._combined_data[self._feature_names].iloc[idx]
            # Compare values, handling NaN equality
            if self._rows_equal(query_values, stored_values):
                return idx

        return None

    def _rows_equal(self, row1: pd.Series, row2: pd.Series) -> bool:
        """Check if two rows are equal, handling NaN values.

        :param row1: First row to compare.
        :type row1: pd.Series
        :param row2: Second row to compare.
        :type row2: pd.Series
        :return: True if rows are equal.
        :rtype: bool
        """
        for col in self._feature_names:
            val1, val2 = row1[col], row2[col]
            # Both NaN -> equal
            if pd.isna(val1) and pd.isna(val2):
                continue
            # One NaN -> not equal
            if pd.isna(val1) or pd.isna(val2):
                return False
            # Compare values
            if val1 != val2:
                # Handle float comparison with tolerance
                if isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
                    if not np.isclose(val1, val2, rtol=1e-9, atol=1e-12):
                        return False
                else:
                    return False
        return True

    def _get_filtered_data(
            self, query_test_data_row: pd.DataFrame) -> pd.DataFrame:
        """Return the filtered data based on the query data.

        :param query_test_data_row: The query data instance.
        :type query_test_data_row: pd.DataFrame
        :return: The filtered dataframe based on the values in query data.
        :rtype: pd.DataFrame
        """
        if not self._should_construct_pandas_query:
            data_copy = self._combined_data
            for column_name, column_data in query_test_data_row.squeeze().items():
                if pd.isnull(column_data):
                    data_copy = data_copy[data_copy[column_name].isna()]
                else:
                    data_copy = data_copy[data_copy[column_name] == column_data]

                if len(data_copy) == 1 or len(data_copy) == 0:
                    # Stop the search if we have found a possible exact match or no match
                    break

            if len(data_copy) == 1:
                # Compare values only, ignoring index differences
                # (query data may have reset index after transformations)
                query_values = query_test_data_row.reset_index(drop=True)
                matched_values = data_copy.filter(self._feature_names).reset_index(drop=True)
                if query_values.equals(matched_values):
                    filtered_df = data_copy
                else:
                    filtered_df = pd.DataFrame(columns=self._combined_data.columns)
            else:
                filtered_df = data_copy
        else:
            queries = []
            for column_name, column_data in query_test_data_row.squeeze().items():
                if pd.isnull(column_data):
                    queries.append("`{}`.isna()".format(column_name))
                elif isinstance(column_data, str):
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

        Uses hash-based lookup for O(1) performance per row instead of
        O(n*m) filtering, where n is dataset size and m is feature count.

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
        for index in range(len(query_test_data)):
            query_row = query_test_data.iloc[index:index + 1]

            # Try fast hash-based lookup first
            matched_idx = self._lookup_by_hash(query_row)

            if matched_idx is not None:
                prediction_output.append(self._combined_data[TARGET].iloc[matched_idx])
            else:
                # Fallback to original filtering method for edge cases
                filtered_df = self._get_filtered_data(query_row)
                prediction_output.append(filtered_df[TARGET].values[0])

        return np.array(prediction_output)

    def __setstate__(self, state):
        """Set the state of the class object so that the wrapped
        model is serializable via pickle.

        :param state: A dictionary of deserialized state.
        :type state: Dict
        """
        self._combined_data = state["_combined_data"]
        self._should_construct_pandas_query = state["_should_construct_pandas_query"]
        self._feature_names = state["_feature_names"]
        # Rebuild hash index after deserialization
        self._row_hash_to_idx = self._build_row_hash_index()

    def __getstate__(self):
        """Return the state so that the wrapped model is
        serializable via pickle.

        Note: The hash index is not serialized as it can be rebuilt
        from the combined_data. This keeps the pickle size smaller.

        :return: The state to be pickled.
        :rtype: Dict
        """
        state = {}
        state["_combined_data"] = self._combined_data
        state["_should_construct_pandas_query"] = self._should_construct_pandas_query
        state["_feature_names"] = self._feature_names
        # Note: _row_hash_to_idx is intentionally not serialized
        # as it is rebuilt in __setstate__
        return state


class PredictionsModelWrapperRegression(PredictionsModelWrapper):
    """Model wrapper to wrap the samples used to train the models
    and the predictions of the model for regression tasks.
    """
    def __init__(self, test_data: pd.DataFrame, y_pred: np.ndarray,
                 should_construct_pandas_query: Optional[bool] = True):
        """Creates a PredictionsModelWrapperRegression object.

        :param test_data: The data that was used to train the model.
        :type test_data: pd.DataFrame
        :param y_pred: Predictions of the model.
        :type y_pred: np.ndarray
        :param should_construct_pandas_query: Whether to use pandas query()
            function to filter the data. If set to False, the data will
            be filtered using iloc(). Defaults to True.
        :type should_construct_pandas_query: Optional[bool]
        """
        super(PredictionsModelWrapperRegression, self).__init__(
            test_data, y_pred, should_construct_pandas_query)


class PredictionsModelWrapperClassification(PredictionsModelWrapper):
    """Model wrapper to wrap the samples used to train the models
    and the predictions of the model for classification tasks.
    """
    def __init__(self, test_data: pd.DataFrame, y_pred: np.ndarray,
                 y_pred_proba: Optional[np.ndarray] = None,
                 should_construct_pandas_query: Optional[bool] = True):
        """Creates a PredictionsModelWrapperClassification object.

        :param test_data: The data that was used to train the model.
        :type test_data: pd.DataFrame
        :param y_pred: Predictions of the model.
        :type y_pred: np.ndarray
        :param y_pred_proba: Prediction probabilities of the model.
        :type y_pred_proba: np.ndarray
        :param should_construct_pandas_query: Whether to use pandas query()
            function to filter the data. If set to False, the data will
            be filtered using iloc(). Defaults to True.
        :type should_construct_pandas_query: Optional[bool]
        """
        super(PredictionsModelWrapperClassification, self).__init__(
            test_data, y_pred, should_construct_pandas_query)
        self._num_classes = None
        if y_pred_proba is not None:
            if not isinstance(y_pred_proba, np.ndarray):
                raise DataValidationException(
                    "Expecting a numpy array for y_pred_proba")

            if len(test_data) != len(y_pred_proba):
                raise DataValidationException(
                    "The number of instances in test data "
                    "do not match with number of prediction probabilities")

            for i in range(0, len(y_pred_proba[0])):
                self._combined_data[
                    TARGET + '_' + str(i)] = y_pred_proba[:, i]
            self._num_classes = len(y_pred_proba[0])

    def predict_proba(self, query_test_data: pd.DataFrame) -> np.ndarray:
        """Return the prediction probabilities based on the query data.

        Uses hash-based lookup for O(1) performance per row.

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
        for index in range(len(query_test_data)):
            query_row = query_test_data.iloc[index:index + 1]

            # Try fast hash-based lookup first
            matched_idx = self._lookup_by_hash(query_row)

            if matched_idx is not None:
                classes_output = []
                for i in range(self._num_classes):
                    classes_output.append(
                        self._combined_data[TARGET + '_' + str(i)].iloc[matched_idx])
                prediction_output.append(classes_output)
            else:
                # Fallback to original filtering method for edge cases
                filtered_df = self._get_filtered_data(query_row)
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
