# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for DatasetWrapper class"""

import numpy as np
import pandas as pd
import pytest
from common_utils import assert_batch_equal, assert_sparse_equal
from ml_wrappers.dataset.dataset_utils import _summarize_data
from ml_wrappers.dataset.dataset_wrapper import DatasetWrapper
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.sparse import csr_matrix

try:
    import torch
except ImportError:
    pass

try:
    import tensorflow as tf
except ImportError:
    pass


@pytest.mark.usefixtures('_clean_dir')
class TestDatasetWrapper(object):
    def test_supported_types(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        wrapper = DatasetWrapper(dataset=test_dataframe)
        df_converted = wrapper.typed_dataset
        assert_frame_equal(df_converted, test_dataframe)

        test_array = test_dataframe.values
        wrapper = DatasetWrapper(dataset=test_array)
        numpy_converted = wrapper.typed_dataset
        assert np.array_equal(numpy_converted, test_array)

        test_series = test_dataframe.squeeze().reset_index(drop=True)
        wrapper = DatasetWrapper(dataset=test_series)
        series_converted = wrapper.typed_dataset
        assert_series_equal(series_converted, test_series,
                            check_names=False)

        sparse_matrix = csr_matrix((3, 4),
                                   dtype=np.int8)
        wrapper = DatasetWrapper(dataset=sparse_matrix)
        sparse_matrix_converted = wrapper.typed_dataset
        assert_sparse_equal(sparse_matrix_converted, sparse_matrix)

        background = _summarize_data(test_dataframe.values)
        DatasetWrapper(dataset=background)

        torch_input = torch.rand(100, 3)
        wrapper = DatasetWrapper(dataset=torch_input)
        torch_converted = wrapper.typed_dataset
        assert torch.all(torch.eq(torch_converted, torch_input))

        tensor_slices = (dict(test_dataframe), None)
        tf_batch_dataset = tf.data.Dataset.from_tensor_slices(tensor_slices).batch(32)
        wrapper = DatasetWrapper(dataset=tf_batch_dataset)
        tf_batch_dataset_converted = wrapper.typed_dataset
        assert_batch_equal(tf_batch_dataset_converted, tf_batch_dataset)

    def test_unsupported_types(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        test_array = test_dataframe.values
        test_list = test_array.tolist()

        with pytest.raises(
                TypeError,
                match='Got type <class \'list\'> which is not supported in DatasetWrapper'):
            DatasetWrapper(test_list)
