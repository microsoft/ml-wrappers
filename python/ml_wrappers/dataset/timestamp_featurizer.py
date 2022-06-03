# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a custom timestamp featurizer for converting timestamp columns to numeric."""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTimestampFeaturizer(BaseEstimator, TransformerMixin):
    """An estimator for featurizing timestamp columns to numeric data.

    :param features: Optional feature column names.
    :type features: list[str]
    :param return_pandas: Whether to return the transformed dataset as a pandas DataFrame.
    :type return_pandas: bool
    :param modify_in_place: Whether to modify the original dataset in place.
    :type modify_in_place: bool
    """

    def __init__(self, features=None, return_pandas=False, modify_in_place=False):
        """Initialize the CustomTimestampFeaturizer.

        :param features: Optional feature column names.
        :type features: list[str]
        :param return_pandas: Whether to return the transformed dataset as a pandas DataFrame.
        :type return_pandas: bool
        :param modify_in_place: Whether to modify the original dataset in place.
        :type modify_in_place: bool
        """
        self.features = features
        self.return_pandas = return_pandas
        self.modify_in_place = modify_in_place
        self._time_col_names = []
        return

    def fit(self, X, y=None):
        """Fits the CustomTimestampFeaturizer.

        :param X: The dataset containing timestamp columns to featurize.
        :type X: numpy.ndarray or pandas.DataFrame or scipy.sparse.csr_matrix
        :param y: The target values.
        :type y: Optional target values (None for unsupervised transformations).
        """
        # If the data was previously successfully summarized, then there are no
        # timestamp columns as it must be numeric.
        # Also, if the dataset is sparse, we can assume there are no timestamps
        if str(type(X)).endswith(".DenseData'>") or issparse(X):
            return self
        tmp_dataset = X
        # If numpy array, temporarily convert to pandas for easier and uniform timestamp handling
        if isinstance(X, np.ndarray):
            tmp_dataset = pd.DataFrame(X, columns=self.features)
        self._time_col_names = [column for column in tmp_dataset.columns if is_datetime(tmp_dataset[column])]
        # Calculate the min date for each column
        self._min = []
        for time_col_name in self._time_col_names:
            self._min.append(tmp_dataset[time_col_name].map(lambda x: x.timestamp()).min())
        return self

    def transform(self, X):
        """Transforms the timestamp columns to numeric type in the given dataset.

        Specifically, extracts the year, month, day, hour, minute, second and time
        since min timestamp in the training dataset.

        :param X: The dataset containing timestamp columns to featurize.
        :type X: numpy.ndarray or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: The transformed dataset.
        :rtype: numpy.ndarray or scipy.sparse.csr_matrix
        """
        tmp_dataset = X
        if len(self._time_col_names) > 0:
            # Temporarily convert to pandas for easier and uniform timestamp handling
            if isinstance(X, np.ndarray):
                tmp_dataset = pd.DataFrame(X, columns=self.features)
            elif not self.modify_in_place:
                # If originally pandas, make a copy to avoid changing the original dataset
                tmp_dataset = X.copy()
            # Get the year, day, month, hour, minute, second
            for idx, time_col_name in enumerate(self._time_col_names):
                tmp_dataset[time_col_name + '_year'] = tmp_dataset[time_col_name].map(lambda x: x.year)
                tmp_dataset[time_col_name + '_month'] = tmp_dataset[time_col_name].map(lambda x: x.month)
                tmp_dataset[time_col_name + '_day'] = tmp_dataset[time_col_name].map(lambda x: x.day)
                tmp_dataset[time_col_name + '_hour'] = tmp_dataset[time_col_name].map(lambda x: x.hour)
                tmp_dataset[time_col_name + '_minute'] = tmp_dataset[time_col_name].map(lambda x: x.minute)
                tmp_dataset[time_col_name + '_second'] = tmp_dataset[time_col_name].map(lambda x: x.second)
                # Replace column itself with difference from min value, leave as same name
                # to keep index so order of other columns remains the same for other transformations
                tmp_dataset[time_col_name] = tmp_dataset[time_col_name].map(lambda x: x.timestamp() - self._min[idx])
            if not self.return_pandas:
                tmp_dataset = tmp_dataset.values
        return tmp_dataset
