# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for DatasetWrapper class"""

import numpy as np
import pandas as pd
import pytest
from ml_wrappers.dataset.dataset_utils import _summarize_data
from ml_wrappers.dataset.dataset_wrapper import DatasetWrapper
from scipy.sparse import csr_matrix

try:
    import torch
except ImportError:
    pass


@pytest.mark.usefixtures('_clean_dir')
class TestDatasetWrapper(object):
    def test_supported_types(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        DatasetWrapper(dataset=test_dataframe)

        test_array = test_dataframe.values
        DatasetWrapper(dataset=test_array)

        test_series = test_dataframe.squeeze()
        DatasetWrapper(dataset=test_series)

        sparse_matrix = csr_matrix((3, 4),
                                   dtype=np.int8)
        DatasetWrapper(dataset=sparse_matrix)

        background = _summarize_data(test_dataframe.values)
        DatasetWrapper(dataset=background)

        torch_input = torch.rand(100, 3)
        DatasetWrapper(dataset=torch_input)

    def test_unsupported_types(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        test_array = test_dataframe.values
        test_list = test_array.tolist()

        with pytest.raises(
                TypeError,
                match='Got type <class \'list\'> which is not not supported in DatasetWrapper'):
            DatasetWrapper(test_list)
