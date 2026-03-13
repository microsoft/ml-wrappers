# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines classes to wrap the training/test data and the corresponding predictions from the model."""

from typing import Optional, Dict, List, Tuple
import hashlib

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

        # Pre-compute numpy arrays and hash index for fast lookups
        self._feature_values = self._combined_data[self._feature_names].values
        self._predictions = y_pred.copy()
        self._row_hash_to_indices = self._build_vectorized_hash_index()

    def _compute_row_hash_fast(self, row: np.ndarray) -> str:
        """Compute a hash for a row using fast string conversion.

        :param row: Row values as numpy array.
        :return: Hash string for the row.
        """
        # Convert to string representation for hashing
        # This handles NaN, various dtypes consistently
        row_str = str(row.tolist())
        return hashlib.md5(row_str.encode()).hexdigest()

    def _build_vectorized_hash_index(self) -> Dict[str, List[int]]:
        """Build hash index using numpy arrays for speed.

        :return: Dictionary mapping row hash to list of indices.
        """
        hash_to_indices: Dict[str, List[int]] = {}
        for idx in range(len(self._feature_values)):
            row_hash = self._compute_row_hash_fast(self._feature_values[idx])
            if row_hash not in hash_to_indices:
                hash_to_indices[row_hash] = []
            hash_to_indices[row_hash].append(idx)
        return hash_to_indices

    def _rows_equal_numpy(self, row1: np.ndarray, row2: np.ndarray) -> bool:
        """Fast row comparison using numpy.

        :param row1: First row values.
        :param row2: Second row values.
        :return: True if rows are equal.
        """
        # Handle NaN: both NaN counts as equal
        nan1 = pd.isna(row1)
        nan2 = pd.isna(row2)
        if not np.array_equal(nan1, nan2):
            return False
        # Compare non-NaN values
        mask = ~nan1
        if not mask.any():
            return True  # All NaN
        # Use array_equal for non-NaN values (handles mixed types)
        try:
            return np.array_equal(row1[mask], row2[mask])
        except (TypeError, ValueError):
            # Fallback for object arrays
            return all(v1 == v2 for v1, v2 in zip(row1[mask], row2[mask]))

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

        # Extract query data as numpy array once (avoid repeated DataFrame ops)
        query_values = query_test_data[self._feature_names].values
        prediction_output = np.empty(len(query_values), dtype=self._predictions.dtype)

        for idx in range(len(query_values)):
            query_row = query_values[idx]
            row_hash = self._compute_row_hash_fast(query_row)

            matched_idx = None
            if row_hash in self._row_hash_to_indices:
                # Check candidates for actual match
                for candidate_idx in self._row_hash_to_indices[row_hash]:
                    if self._rows_equal_numpy(query_row, self._feature_values[candidate_idx]):
                        matched_idx = candidate_idx
                        break

            if matched_idx is not None:
                prediction_output[idx] = self._predictions[matched_idx]
            else:
                # Fallback to original filtering method for edge cases
                query_row_df = query_test_data.iloc[idx:idx + 1]
                filtered_df = self._get_filtered_data(query_row_df)
                prediction_output[idx] = filtered_df[TARGET].values[0]

        return prediction_output

    def __setstate__(self, state):
        """Set the state of the class object so that the wrapped
        model is serializable via pickle.

        :param state: A dictionary of deserialized state.
        :type state: Dict
        """
        self._combined_data = state["_combined_data"]
        self._should_construct_pandas_query = state["_should_construct_pandas_query"]
        self._feature_names = state["_feature_names"]
        # Rebuild numpy arrays and hash index after deserialization
        self._feature_values = self._combined_data[self._feature_names].values
        self._predictions = self._combined_data[TARGET].values
        self._row_hash_to_indices = self._build_vectorized_hash_index()

    def __getstate__(self):
        """Return the state so that the wrapped model is
        serializable via pickle.

        Note: The hash index and numpy arrays are not serialized as they
        can be rebuilt from the combined_data. This keeps the pickle size smaller.

        :return: The state to be pickled.
        :rtype: Dict
        """
        state = {}
        state["_combined_data"] = self._combined_data
        state["_should_construct_pandas_query"] = self._should_construct_pandas_query
        state["_feature_names"] = self._feature_names
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
            # Store proba as numpy array for fast access
            self._predictions_proba = y_pred_proba.copy()
        else:
            self._predictions_proba = None

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

        # Extract query data as numpy array once
        query_values = query_test_data[self._feature_names].values
        prediction_output = np.empty((len(query_values), self._num_classes), dtype=np.float64)

        for idx in range(len(query_values)):
            query_row = query_values[idx]
            row_hash = self._compute_row_hash_fast(query_row)

            matched_idx = None
            if row_hash in self._row_hash_to_indices:
                for candidate_idx in self._row_hash_to_indices[row_hash]:
                    if self._rows_equal_numpy(query_row, self._feature_values[candidate_idx]):
                        matched_idx = candidate_idx
                        break

            if matched_idx is not None:
                prediction_output[idx] = self._predictions_proba[matched_idx]
            else:
                # Fallback to original filtering method
                query_row_df = query_test_data.iloc[idx:idx + 1]
                filtered_df = self._get_filtered_data(query_row_df)
                for i in range(self._num_classes):
                    prediction_output[idx, i] = filtered_df[TARGET + '_' + str(i)].values[0]

        return prediction_output

    def __setstate__(self, state):
        """Set the state of the class object so that the wrapped
        model is serializable via pickle.

        :param state: A dictionary of deserialized state.
        :type state: Dict
        """
        super().__setstate__(state)
        self._num_classes = state["_num_classes"]
        # Rebuild predictions_proba from combined_data
        if self._num_classes is not None:
            proba_cols = [TARGET + '_' + str(i) for i in range(self._num_classes)]
            self._predictions_proba = self._combined_data[proba_cols].values
        else:
            self._predictions_proba = None

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
