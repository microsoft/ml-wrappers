# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for DatasetWrapper class"""

import gc
import sys
import weakref
from types import SimpleNamespace

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

    @pytest.mark.parametrize('sparse', [False, True])
    @pytest.mark.parametrize('max_dimensions', [2, 4, 6])
    def test_reduce_examples_preserves_original_data(self, sparse, max_dimensions):
        dataset = np.random.RandomState(7).normal(size=(12, 4))
        original = csr_matrix(dataset) if sparse else dataset.copy()
        wrapper = DatasetWrapper(original)

        reduced = wrapper._reduce_examples(max_dim_clustering=max_dimensions)

        assert reduced.shape == (12, min(4, max_dimensions))
        if max_dimensions < 4:
            assert np.isfinite(reduced).all()
            assert np.var(reduced, axis=0).sum() > 0
        else:
            assert reduced is original
        if sparse:
            assert_sparse_equal(wrapper.dataset, csr_matrix(dataset))
        else:
            np.testing.assert_array_equal(wrapper.dataset, dataset)

    @pytest.mark.parametrize('sparse', [False, True])
    def test_sample_with_real_kmeans(self, sparse):
        dataset = np.random.RandomState(7).normal(size=(200, 4))
        wrapper = DatasetWrapper(csr_matrix(dataset) if sparse else dataset.copy())

        sampled = wrapper.sample(max_dim_clustering=2, sampling_method='kmeans')

        sampled_values = sampled.toarray() if sparse else sampled
        assert sampled_values.shape == (100, 4)
        original_rows = set(map(tuple, dataset))
        assert all(tuple(row) in original_rows for row in sampled_values)
        original = wrapper.original_dataset.toarray() if sparse else wrapper.original_dataset
        np.testing.assert_array_equal(original, dataset)

    @pytest.mark.parametrize(('scores', 'expected_clusters'), [
        ([0.8, 0.2], 256),
        ([0.2, 0.8], 128),
        ([np.nan, np.nan], 128)
    ])
    def test_kmeans_selects_best_silhouette_score(self, monkeypatch, scores, expected_clusters):
        dataset = np.arange(1024).reshape(512, 2)
        wrapper = DatasetWrapper(dataset)
        candidates = []
        score_iterator = iter(scores)

        class FakeKMeans:
            def __init__(self, n_clusters):
                candidates.append(n_clusters)

            def fit(self, examples):
                np.testing.assert_array_equal(examples, dataset)
                self.labels_ = np.arange(len(examples)) % candidates[-1]
                return self

        def silhouette_score(examples, labels):
            np.testing.assert_array_equal(examples, dataset)
            assert len(labels) == len(dataset)
            return next(score_iterator)

        monkeypatch.setattr('sklearn.cluster.KMeans', FakeKMeans)
        monkeypatch.setattr('sklearn.metrics.silhouette_score', silhouette_score)

        assert wrapper._find_k_kmeans() == expected_clusters
        assert candidates == [256, 128]

    @pytest.mark.parametrize('num_rows', [6, 20])
    def test_hdbscan_sampling_count_is_capped(self, monkeypatch, num_rows):
        dataset = np.arange(num_rows * 2).reshape(num_rows, 2)
        wrapper = DatasetWrapper(dataset)

        class FakeHDBSCAN:
            def __init__(self, min_cluster_size):
                assert min_cluster_size == 2

            def fit(self, examples):
                np.testing.assert_array_equal(examples, dataset)
                self.labels_ = np.arange(num_rows) % 2
                return self

        monkeypatch.setitem(sys.modules, 'hdbscan', SimpleNamespace(HDBSCAN=FakeHDBSCAN))

        assert wrapper._find_k_hdbscan() == min(10, num_rows)

    def test_sample_falls_back_when_hdbscan_fails(self, monkeypatch, caplog):
        dataset = np.random.RandomState(7).normal(size=(200, 2))
        wrapper = DatasetWrapper(dataset.copy())

        def unavailable_hdbscan(max_dim_clustering):
            raise ImportError('hdbscan unavailable')

        monkeypatch.setattr(wrapper, '_find_k_hdbscan', unavailable_hdbscan)

        sampled = wrapper.sample()

        assert sampled.shape == (100, 2)
        assert 'hdbscan unavailable' in caplog.text
        original_rows = set(map(tuple, dataset))
        assert all(tuple(row) in original_rows for row in sampled)
        np.testing.assert_array_equal(wrapper.original_dataset, dataset)

    def test_sample_keeps_data_when_cluster_count_reaches_row_count(self, monkeypatch):
        dataset = np.arange(400).reshape(200, 2)
        wrapper = DatasetWrapper(dataset)
        monkeypatch.setattr(wrapper, '_find_k_hdbscan', lambda max_dim_clustering: 200)

        assert wrapper.sample() is dataset
        np.testing.assert_array_equal(wrapper.dataset, dataset)

    @pytest.mark.parametrize('clear_references', [False, True])
    def test_clear_releases_references_only_when_requested(self, clear_references):
        dataset = pd.DataFrame({'category': [0, 1], 'value': [10, 20]})
        wrapper = DatasetWrapper(dataset, clear_references=clear_references)
        dataset_reference = weakref.ref(dataset)
        encoder_reference = weakref.ref(wrapper.one_hot_encode([0]))
        wrapper.compute_summary()
        transformed_reference = weakref.ref(wrapper.dataset)
        del dataset

        wrapper._clear()
        gc.collect()

        if clear_references:
            assert dataset_reference() is None
            assert encoder_reference() is None
            assert transformed_reference() is None
            assert wrapper.dataset is None
            assert wrapper.original_dataset is None
            assert wrapper.original_dataset_with_type is None
            assert wrapper.summary_dataset is None
        else:
            assert dataset_reference() is wrapper.original_dataset_with_type
            assert encoder_reference() is not None
            assert transformed_reference() is wrapper.dataset
            np.testing.assert_array_equal(wrapper.dataset, [[1, 0, 10], [0, 1, 20]])
