# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines helpful utilities for the DatasetWrapper."""

import logging

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import vstack as sparse_vstack
from sklearn.utils import shuffle
from sklearn.utils.sparsefuncs import csc_median_axis_0

from ..common.gpu_kmeans import kmeans
from ..common.warnings_suppressor import shap_warnings_suppressor

with shap_warnings_suppressor():
    try:
        import shap
        shap_installed = True
    except BaseException:
        shap_installed = False

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def _generate_augmented_data(x, max_num_of_augmentations=np.inf):
    """Augment x by appending x with itself shuffled columnwise many times.

    :param x: data that has to be augmented, array or sparse matrix of 2 dimensions
    :type x: numpy.array or scipy.sparse.csr_matrix
    :param max_augment_data_size: number of times we stack permuted x to augment.
    :type max_augment_data_size: int
    :return: augmented data with roughly number of rows that are equal to number of columns
    :rtype: numpy.array or scipy.sparse.csr_matrix
    """
    x_augmented = x
    vstack = sparse_vstack if issparse(x) else np.vstack
    for i in range(min(x.shape[1] // x.shape[0] - 1, max_num_of_augmentations)):
        x_permuted = shuffle(x.T, random_state=i).T
        x_augmented = vstack([x_augmented, x_permuted])

    return x_augmented


def _summarize_data(X, k=10, use_gpu=False, to_round_values=True):
    """Summarize a dataset.

    For dense dataset, use k mean samples weighted by the number of data points they
    each represent.
    For sparse dataset, use a sparse row for the background with calculated
    median for dense columns.

    :param X: Matrix of data samples to summarize (# samples x # features).
    :type X: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
    :param k: Number of cluster centroids to use for approximation.
    :type k: int
    :param to_round_values: When using kmeans, for each element of every cluster centroid to match the nearest value
        from X in the corresponding dimension. This ensures discrete features
        always get a valid value.  Ignored for sparse data sample.
    :type to_round_values: bool
    :return: summarized numpy array or csr_matrix object.
    :rtype: numpy.array or scipy.sparse.csr_matrix or DenseData
    """
    is_sparse = issparse(X)
    if not str(type(X)).endswith(".DenseData'>"):
        if is_sparse:
            module_logger.debug('Creating sparse data summary as csr matrix')
            # calculate median of sparse background data
            median_dense = csc_median_axis_0(X.tocsc())
            return csr_matrix(median_dense)
        elif len(X) > 10 * k:
            module_logger.debug('Create dense data summary with k-means')
            # use kmeans to summarize the examples for initialization
            # if there are more than 10 x k of them
            if use_gpu:
                return kmeans(X, k, to_round_values)
            else:
                if not shap_installed:
                    raise RuntimeError('shap is required to compute dataset summary in DatasetWrapper')
                return shap.kmeans(X, k, to_round_values)
    return X
