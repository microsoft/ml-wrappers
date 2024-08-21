# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function"""

import sys

import pandas as pd
import pytest
from common_utils import (create_catboost_classifier,
                          create_catboost_regressor,
                          create_fastai_tabular_classifier,
                          create_fastai_tabular_classifier_multimetric,
                          create_fastai_tabular_regressor,
                          create_keras_classifier, create_keras_regressor,
                          create_lightgbm_classifier,
                          create_lightgbm_regressor,
                          create_pytorch_multiclass_classifier,
                          create_pytorch_regressor,
                          create_scikit_keras_multiclass_classifier,
                          create_scikit_keras_regressor,
                          create_sklearn_linear_regressor,
                          create_sklearn_logistic_regressor, create_tf_model,
                          create_xgboost_classifier, create_xgboost_regressor)
from constants import DatasetConstants
from ml_wrappers import wrap_model
from ml_wrappers.dataset.dataset_wrapper import DatasetWrapper
from train_wrapper_utils import (train_classification_model_numpy,
                                 train_classification_model_pandas,
                                 train_regression_model_numpy,
                                 train_regression_model_pandas)
from wrapper_validator import validate_wrapped_regression_model

try:
    import tensorflow as tf
except ImportError:
    pass


@pytest.mark.usefixtures('_clean_dir')
class TestModelWrapper(object):
    def test_wrap_sklearn_logistic_regression_model(self, iris):
        train_classification_model_numpy(
            create_sklearn_logistic_regressor, iris)
        train_classification_model_pandas(
            create_sklearn_logistic_regressor, iris)
        train_classification_model_numpy(
            create_sklearn_logistic_regressor, iris,
            use_dataset_wrapper=False)
        train_classification_model_pandas(
            create_sklearn_logistic_regressor, iris,
            use_dataset_wrapper=False)

    def test_wrap_pytorch_classification_model(self, iris):
        train_classification_model_numpy(
            create_pytorch_multiclass_classifier, iris)
        train_classification_model_numpy(
            create_pytorch_multiclass_classifier, iris,
            use_dataset_wrapper=False)

    def test_wrap_xgboost_classification_model(self, iris):
        train_classification_model_numpy(create_xgboost_classifier, iris)
        train_classification_model_pandas(create_xgboost_classifier, iris)

    def test_wrap_catboost_classification_model(self, iris):
        train_classification_model_numpy(create_catboost_classifier, iris)
        train_classification_model_pandas(create_catboost_classifier, iris)

    def test_wrap_lightgbm_classification_model(self, iris):
        train_classification_model_numpy(create_lightgbm_classifier, iris)
        train_classification_model_pandas(create_lightgbm_classifier, iris)

    def test_wrap_keras_classification_model(self, iris):
        train_classification_model_numpy(create_keras_classifier, iris)
        train_classification_model_pandas(create_keras_classifier, iris)

    def test_wrap_scikit_keras_classification_model(self, iris):
        train_classification_model_numpy(create_scikit_keras_multiclass_classifier, iris)
        train_classification_model_pandas(create_scikit_keras_multiclass_classifier, iris)

    # Skip for older versions due to latest fastai not supporting 3.6
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Fastai not supported for older versions')
    # Skip is using macos due to fastai failing on latest macos
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='Fastai not supported for latest macos')
    def test_wrap_fastai_classification_model(self, iris):
        train_classification_model_pandas(create_fastai_tabular_classifier, iris)

    # Skip for older versions due to latest fastai not supporting 3.6
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Fastai not supported for older versions')
    # Skip is using macos due to fastai failing on latest macos
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='Fastai not supported for latest macos')
    def test_wrap_fastai_classification_model_multimetric(self, iris):
        iris = iris.copy()
        data_to_transform = [DatasetConstants.Y_TRAIN, DatasetConstants.Y_TEST]
        for data in data_to_transform:
            iris[data][iris[data] == 2] = 1
        train_classification_model_pandas(
            create_fastai_tabular_classifier_multimetric, iris,
            validate_single_row=True)

    def test_wrap_sklearn_linear_regression_model(self, housing):
        train_regression_model_numpy(
            create_sklearn_linear_regressor, housing)
        train_regression_model_pandas(
            create_sklearn_linear_regressor, housing)
        train_regression_model_numpy(
            create_sklearn_linear_regressor, housing,
            use_dataset_wrapper=False)
        train_regression_model_pandas(
            create_sklearn_linear_regressor, housing,
            use_dataset_wrapper=False)

    def test_wrap_pytorch_regression_model(self, housing):
        train_regression_model_numpy(
            create_pytorch_regressor, housing)

    def test_wrap_xgboost_regression_model(self, housing):
        train_regression_model_numpy(create_xgboost_regressor, housing)
        train_regression_model_pandas(create_xgboost_regressor, housing)

    def test_wrap_catboost_regression_model(self, housing):
        train_regression_model_numpy(create_catboost_regressor, housing)
        train_regression_model_pandas(create_catboost_regressor, housing)

    def test_wrap_lightgbm_regression_model(self, housing):
        train_regression_model_numpy(create_lightgbm_regressor, housing)
        train_regression_model_pandas(create_lightgbm_regressor, housing)

    def test_wrap_keras_regression_model(self, housing):
        train_regression_model_numpy(create_keras_regressor, housing)
        train_regression_model_pandas(create_keras_regressor, housing)

    def test_wrap_scikit_keras_regression_model(self, housing):
        train_regression_model_numpy(create_scikit_keras_regressor, housing)
        train_regression_model_pandas(create_scikit_keras_regressor, housing)

    # Skip for older versions due to latest fastai not supporting 3.6
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Fastai not supported for older versions')
    # Skip is using macos due to fastai failing on latest macos
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='Fastai not supported for latest macos')
    def test_wrap_fastai_regression_model(self, iris):
        train_regression_model_pandas(create_fastai_tabular_regressor, iris)

    def test_batch_dataset(self, housing):
        X_train = housing[DatasetConstants.X_TRAIN]
        X_test = housing[DatasetConstants.X_TEST]
        y_train = housing[DatasetConstants.Y_TRAIN]
        y_test = housing[DatasetConstants.Y_TEST]
        features = housing[DatasetConstants.FEATURES]
        X_train_df = pd.DataFrame(X_train, columns=list(features))
        X_test_df = pd.DataFrame(X_test, columns=list(features))
        inp = (dict(X_train_df), y_train)
        inp_ds = tf.data.Dataset.from_tensor_slices(inp).batch(32)
        val = (dict(X_test_df), y_test)
        val_ds = tf.data.Dataset.from_tensor_slices(val).batch(32)
        model = create_tf_model(inp_ds, val_ds, features)
        wrapped_dataset = DatasetWrapper(val_ds)
        wrapped_model = wrap_model(model, wrapped_dataset, model_task='regression')
        validate_wrapped_regression_model(wrapped_model, val_ds)
