# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pandas as pd
import pytest
from ml_wrappers.dataset import CustomTimestampFeaturizer
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from rai_test_utils.datasets.tabular import create_timeseries_data

from constants import DatasetConstants


@pytest.mark.usefixtures('_clean_dir')
class TestTimestampFeaturizer(object):

    def test_working(self):
        assert True

    def test_no_timestamps(self, iris):
        # create pandas dataframes without any timestamps
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        featurizer = CustomTimestampFeaturizer(iris[DatasetConstants.FEATURES]).fit(x_train)
        result = featurizer.transform(x_test)
        # Assert result is same as before, pandas dataframe
        assert(isinstance(result, pd.DataFrame))
        # Assert the result is the same as the original passed in data (no featurization was done)
        assert(result.equals(x_test))

    @pytest.mark.parametrize(("sample_cnt_per_grain", "grains_dict"), [
        (240, {}),
        (20, {'fruit': ['apple', 'grape'], 'store': [100, 200, 50]})])
    def test_timestamp_featurization(self, sample_cnt_per_grain, grains_dict):
        # create timeseries data
        X, _ = create_timeseries_data(sample_cnt_per_grain, 'time', 'y', grains_dict)
        original_cols = list(X.columns.values)
        # featurize and validate the timestamp column
        featurizer = CustomTimestampFeaturizer(original_cols).fit(X)
        result = featurizer.transform(X)
        # Form a temporary dataframe for validation
        tmp_result = pd.DataFrame(result)
        # Assert there are no timestamp columns
        assert([column for column in tmp_result.columns if is_datetime(tmp_result[column])] == [])
        # Assert we have the expected number of columns - 1 time columns * 6 featurized plus original
        assert(result.shape[1] == len(original_cols) + 6)

    @pytest.mark.parametrize(("return_pandas"), [True, False])
    def test_separate_fit_with_no_features(self, return_pandas):
        sample_cnt_per_grain = 20
        grains_dict = {'fruit': ['apple', 'grape'], 'store': [100, 200, 50]}
        # create timeseries data
        X, _ = create_timeseries_data(sample_cnt_per_grain, 'time', 'y', grains_dict)
        original_cols = list(X.columns.values)
        # featurize and validate the timestamp column as a separate fit call and fit_transform
        # Note: in this case we don't pass the feature names to the constructor
        ctf1 = CustomTimestampFeaturizer(return_pandas=return_pandas)
        ctf2 = CustomTimestampFeaturizer(return_pandas=return_pandas)
        ctf1.fit(X)
        result1 = ctf1.transform(X)
        result2 = ctf2.fit_transform(X)
        for result in [result1, result2]:
            if not return_pandas:
                assert not isinstance(result, pd.DataFrame)
                # Form a temporary dataframe for validation
                result = pd.DataFrame(result)
            # Assert there are no timestamp columns
            assert([column for column in result.columns if is_datetime(result[column])] == [])
            # Assert we have the expected number of columns - 1 time columns * 6 featurized plus original
            assert(result.shape[1] == len(original_cols) + 6)
