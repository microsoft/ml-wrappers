# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for DatasetWrapper class"""

import sys

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

    # Skip on macOS due to TensorFlow batch dataset iterator hanging
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='TensorFlow batch dataset hangs on macOS')
    def test_tf_batch_dataset(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
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

    def test_string_index_supports_pandas_mapping_apis(self, monkeypatch):
        """Test current and legacy Pandas element-wise mapping APIs."""
        dataset = np.array([['a', 1], ['b', 2]], dtype=object)
        wrapper = DatasetWrapper(dataset.copy())

        assert wrapper.string_index() is not None
        np.testing.assert_array_equal(
            wrapper.dataset[:, 0], np.array([0.0, 1.0], dtype=object))

        def legacy_applymap(dataframe, function):
            return dataframe.apply(lambda column: column.map(function))

        monkeypatch.delattr(pd.DataFrame, 'map', raising=False)
        monkeypatch.setattr(
            pd.DataFrame, 'applymap', legacy_applymap, raising=False)
        fallback_wrapper = DatasetWrapper(dataset.copy())

        assert fallback_wrapper.string_index() is not None
        np.testing.assert_array_equal(
            fallback_wrapper.dataset[:, 0],
            np.array([0.0, 1.0], dtype=object))

    def test_typed_wrapper_preserves_dataframe_columns_and_dtypes(self):
        dataset = pd.DataFrame({
            'count': pd.Series([1, 2], dtype='int32'),
            'value': pd.Series([1.5, 2.5], dtype='float64'),
            'category': pd.Categorical(['a', 'b'])
        })
        wrapper = DatasetWrapper(dataset)

        assert_frame_equal(wrapper.typed_dataset, dataset)
        assert_frame_equal(
            wrapper.typed_wrapper_func(wrapper.dataset[1]),
            dataset.iloc[[1]].reset_index(drop=True))

    @pytest.mark.parametrize('index', [
        pd.RangeIndex(2),
        pd.Index(['first', 'second'], name='row'),
        pd.MultiIndex.from_tuples(
            [('a', 1), ('b', 2)], names=['group', 'row'])
    ])
    def test_reset_and_set_index(self, index):
        dataset = pd.DataFrame({'value': [10, 20]}, index=index)
        wrapper = DatasetWrapper(dataset)

        wrapper.reset_index()

        assert_frame_equal(wrapper.typed_dataset, dataset)
        expected = dataset.reset_index() if not isinstance(index, pd.RangeIndex) else dataset
        expected = expected[['value'] + [column for column in expected.columns if column != 'value']]
        assert_frame_equal(
            wrapper.typed_wrapper_func(wrapper.dataset, keep_index_as_feature=True),
            expected)

        wrapper.set_index()

        np.testing.assert_array_equal(wrapper.dataset, dataset.values)
        assert wrapper.get_features() == ['value']

    @pytest.mark.parametrize(('features', 'subset', 'expected'), [
        (None, None, ['a', 'b', 'c']),
        (None, [2, 0], ['c', 'a']),
        (['x', 'y', 'z'], None, ['x', 'y', 'z']),
        (['x', 'y', 'z'], [2, 0], ['z', 'x'])
    ])
    def test_get_features(self, features, subset, expected):
        wrapper = DatasetWrapper(pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c']))

        assert wrapper.get_features(features, explain_subset=subset) == expected

    @pytest.mark.parametrize('sparse', [False, True])
    def test_take_subset_preserves_original_data(self, sparse):
        dataset = np.arange(12).reshape(3, 4)
        original = csr_matrix(dataset) if sparse else dataset
        wrapper = DatasetWrapper(original)

        wrapper.take_subset([3, 1])
        wrapper.take_subset([0])

        if sparse:
            assert_sparse_equal(wrapper.dataset.sorted_indices(), csr_matrix(dataset[:, [3, 1]]))
            assert_sparse_equal(wrapper.original_dataset, csr_matrix(dataset))
        else:
            np.testing.assert_array_equal(wrapper.dataset, dataset[:, [3, 1]])
            np.testing.assert_array_equal(wrapper.original_dataset, dataset)
        assert wrapper.original_dataset_with_type is original

    def test_string_index_selected_columns(self):
        dataset = np.array([['b', 10, 'x'], ['a', 20, 'y']], dtype=object)
        wrapper = DatasetWrapper(dataset.copy())

        indexer = wrapper.string_index(columns=[0])

        np.testing.assert_array_equal(
            wrapper.dataset, np.array([[1, 10, 'x'], [0, 20, 'y']], dtype=object))
        assert wrapper.string_index(columns=[2]) is indexer

    @pytest.mark.parametrize(('bucket_unknown', 'expected'), [
        (False, [1, 3, 4, 3]),
        (True, [1, 3, 3, 3])
    ])
    def test_apply_indexer_handles_unknown_categories(self, bucket_unknown, expected):
        background = DatasetWrapper(np.array([['blue', 1], ['red', 2]], dtype=object))
        indexer = background.string_index()
        wrapper = DatasetWrapper(np.array([
            ['red', 10], ['green', 20], ['yellow', 30], ['green', 40]
        ], dtype=object))

        wrapper.apply_indexer(indexer, bucket_unknown=bucket_unknown)
        wrapper.apply_indexer(indexer, bucket_unknown=bucket_unknown)

        np.testing.assert_array_equal(wrapper.dataset[:, 0], expected)
        np.testing.assert_array_equal(wrapper.dataset[:, 1], [10, 20, 30, 40])
        assert np.issubdtype(wrapper.dataset.dtype, np.floating)

    def test_one_hot_encode_and_apply_encoder(self):
        wrapper = DatasetWrapper(np.array([[10, 0], [20, 1], [30, 0]]))

        encoder = wrapper.one_hot_encode([1])

        np.testing.assert_array_equal(
            wrapper.dataset, [[1, 0, 10], [0, 1, 20], [1, 0, 30]])
        assert wrapper.one_hot_encode([1]) is encoder
        np.testing.assert_array_equal(
            wrapper.dataset, [[1, 0, 10], [0, 1, 20], [1, 0, 30]])

        evaluation = DatasetWrapper(np.array([[40, 2], [50, 1]]))
        evaluation.apply_one_hot_encoder(encoder)
        evaluation.apply_one_hot_encoder(encoder)

        np.testing.assert_array_equal(evaluation.dataset, [[0, 0, 40], [0, 1, 50]])

    def test_sparse_encoding_is_unchanged(self):
        dataset = csr_matrix([[0, 1], [2, 0]])
        wrapper = DatasetWrapper(dataset)

        assert wrapper.string_index() is None
        assert wrapper.one_hot_encode([1]) is None
        assert wrapper.timestamp_featurizer() is None
        assert_sparse_equal(wrapper.dataset, dataset)

    def test_timestamp_featurizer_and_apply(self):
        dataset = pd.DataFrame({
            'time': pd.to_datetime(['2020-01-01', '2020-01-02']),
            'value': [10, 20]
        })
        wrapper = DatasetWrapper(dataset)

        featurizer = wrapper.timestamp_featurizer()

        expected = [
            [0, 10, 2020, 1, 1, 0, 0, 0],
            [86400, 20, 2020, 1, 2, 0, 0, 0]
        ]
        np.testing.assert_array_equal(wrapper.dataset, expected)
        assert wrapper.timestamp_featurizer() is featurizer
        np.testing.assert_array_equal(wrapper.dataset, expected)

        evaluation = DatasetWrapper(pd.DataFrame({
            'time': pd.to_datetime(['2020-01-03']), 'value': [30]
        }))
        evaluation.apply_timestamp_featurizer(featurizer)
        evaluation.apply_timestamp_featurizer(featurizer)

        np.testing.assert_array_equal(
            evaluation.dataset, [[172800, 30, 2020, 1, 3, 0, 0, 0]])

    def test_compute_sparse_summary_and_subset(self):
        dataset = csr_matrix([[0, 2, 4], [0, 4, 0], [6, 8, 2]])
        wrapper = DatasetWrapper(dataset)

        wrapper.compute_summary()

        assert_sparse_equal(wrapper.dataset, csr_matrix([[0, 4, 2]], dtype=float))
        assert_sparse_equal(wrapper.summary_dataset, csr_matrix([[0, 4, 2]], dtype=float))
        assert_sparse_equal(wrapper.original_dataset, dataset)

        wrapper.take_subset([2, 0])
        wrapper.compute_summary()

        assert_sparse_equal(wrapper.dataset, csr_matrix([[2, 0]], dtype=float))
        assert_sparse_equal(wrapper.summary_dataset, csr_matrix([[0, 4, 2]], dtype=float))

    def test_compute_dense_summary(self):
        dataset = np.tile([[0., 2.], [2., 4.], [4., 6.]], (4, 1))
        wrapper = DatasetWrapper(dataset)

        wrapper.compute_summary(nclusters=1)

        np.testing.assert_array_equal(wrapper.dataset.data, [[2., 4.]])
        np.testing.assert_array_equal(wrapper.summary_dataset.data, [[2., 4.]])
        np.testing.assert_array_equal(wrapper.original_dataset, dataset)

        wrapper.compute_summary(nclusters=2)

        np.testing.assert_array_equal(wrapper.dataset.data, [[2., 4.]])

    @pytest.mark.parametrize('sparse', [False, True])
    @pytest.mark.parametrize('max_augmentations', [0, 1])
    def test_augment_data_respects_limit(self, sparse, max_augmentations):
        dataset = np.arange(16).reshape(2, 8)
        wrapper = DatasetWrapper(csr_matrix(dataset) if sparse else dataset)

        wrapper.augment_data(max_num_of_augmentations=max_augmentations)

        augmented = wrapper.dataset.toarray() if sparse else wrapper.dataset
        assert augmented.shape == (2 * (max_augmentations + 1), 8)
        np.testing.assert_array_equal(augmented[:2], dataset)
        if max_augmentations:
            np.testing.assert_array_equal(np.sort(augmented[2:], axis=1), dataset)

    def test_sample_small_dataset_is_unchanged(self, monkeypatch):
        dataset = np.arange(398).reshape(199, 2)
        wrapper = DatasetWrapper(dataset)

        def unexpected_clustering(*args):
            pytest.fail('Small datasets should not require clustering')

        monkeypatch.setattr(wrapper, '_find_k_hdbscan', unexpected_clustering)
        monkeypatch.setattr(wrapper, '_find_k_kmeans', unexpected_clustering)

        np.testing.assert_array_equal(wrapper.sample(), dataset)
        np.testing.assert_array_equal(wrapper.dataset, dataset)

    @pytest.mark.parametrize('num_rows', [200, 10001])
    def test_sample_uses_cluster_count(self, monkeypatch, num_rows):
        dataset = np.arange(num_rows * 2).reshape(num_rows, 2)
        wrapper = DatasetWrapper(dataset)

        def find_k(max_dim_clustering):
            assert wrapper.dataset.shape == (min(num_rows, 10000), 2)
            assert max_dim_clustering == 1
            return 5

        monkeypatch.setattr(wrapper, '_find_k_kmeans', find_k)

        sampled = wrapper.sample(max_dim_clustering=1, sampling_method='kmeans')

        assert sampled.shape == (5, 2)
        original_rows = set(map(tuple, dataset))
        assert all(tuple(row) in original_rows for row in sampled)
        np.testing.assert_array_equal(wrapper.dataset, sampled)
        np.testing.assert_array_equal(wrapper.original_dataset, dataset)
