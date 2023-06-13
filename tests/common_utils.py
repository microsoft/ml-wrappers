# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Defines common utilities for explanations
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn import linear_model, svm
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (fetch_20newsgroups, fetch_california_housing,
                              load_breast_cancer, load_diabetes, load_iris,
                              load_wine, make_classification)
from rai_test_utils.utilities import retrieve_dataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    pass

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    pass

try:
    from tensorflow import keras
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Activation, Dense, Dropout, concatenate
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.wrappers.scikit_learn import (KerasClassifier,
                                                        KerasRegressor)
except (ImportError, TypeError):
    pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

try:
    from fastai.callback.core import Callback
    from fastai.data.block import CategoryBlock, RegressionBlock
    from fastai.metrics import accuracy
    from fastai.tabular.data import TabularDataLoaders
    from fastai.tabular.learner import tabular_learner
except (ImportError, SyntaxError):
    Callback = object
    # Skip for older versions of python due to breaking changes in fastai
    pass

from datasets_utils import retrieve_dataset
from pandas import read_csv

LIGHTGBM_METHOD = 'mimic.lightgbm'
LINEAR_METHOD = 'mimic.linear'
SGD_METHOD = 'mimic.sgd'
TREE_METHOD = 'mimic.tree'
LABEL = 'label'


def get_mimic_method(surrogate_model):
    from interpret_community.mimic.models.lightgbm_model import \
        LGBMExplainableModel
    from interpret_community.mimic.models.linear_model import (
        LinearExplainableModel, SGDExplainableModel)
    from interpret_community.mimic.models.tree_model import \
        DecisionTreeExplainableModel
    if surrogate_model == LGBMExplainableModel:
        return LIGHTGBM_METHOD
    elif surrogate_model == LinearExplainableModel:
        return LINEAR_METHOD
    elif surrogate_model == SGDExplainableModel:
        return SGD_METHOD
    elif surrogate_model == DecisionTreeExplainableModel:
        return TREE_METHOD
    else:
        raise Exception("Unsupported surrogate model")


def create_binary_sparse_newsgroups_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']
    x_train = newsgroups_train.data
    x_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_validation = newsgroups_test.target
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=2**16)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    return x_train, x_test, y_train, y_validation, class_names, vectorizer


def create_multiclass_sparse_newsgroups_data():
    remove = ('headers', 'footers', 'quotes')
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    from sklearn.datasets import fetch_20newsgroups
    ngroups = fetch_20newsgroups(subset='train', categories=categories,
                                 shuffle=True, random_state=42, remove=remove)
    x_train, x_test, y_train, y_validation = train_test_split(ngroups.data, ngroups.target,
                                                              test_size=0.02, random_state=42)
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=2**16)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    return x_train, x_test, y_train, y_validation, categories, vectorizer


def create_random_forest_tfidf():
    vectorizer = TfidfVectorizer(lowercase=False)
    rf = RandomForestClassifier(n_estimators=500, random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('rf', rf)])


def create_random_forest_vectorizer():
    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)
    rf = RandomForestClassifier(n_estimators=500, random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('rf', rf)])


def create_logistic_vectorizer():
    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)
    lr = LogisticRegression(random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('lr', lr)])


def create_linear_vectorizer():
    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)
    lr = LinearRegression()
    return Pipeline([('vectorizer', vectorizer), ('lr', lr)])


def create_sklearn_random_forest_classifier(X, y):
    rfc = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=777)
    model = rfc.fit(X, y)
    return model


def create_lightgbm_classifier(X, y):
    lgbm = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1,
                          max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    model = lgbm.fit(X, y)
    return model


def create_lightgbm_regressor(X, y):
    lgbm = LGBMRegressor(boosting_type='gbdt', learning_rate=0.1,
                         max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    model = lgbm.fit(X, y)
    return model


def create_xgboost_classifier(X, y):
    xgb = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100,
                        n_jobs=1, random_state=777)
    model = xgb.fit(X, y)
    return model


def create_xgboost_regressor(X, y):
    xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100,
                       n_jobs=1, random_state=777)
    model = xgb.fit(X, y)
    return model


def create_catboost_classifier(X, y):
    xgb = CatBoostClassifier(random_state=777, logging_level='Silent')
    model = xgb.fit(X, y)
    return model


def create_catboost_regressor(X, y):
    xgb = CatBoostRegressor(random_state=777, logging_level='Silent')
    model = xgb.fit(X, y)
    return model


def create_sklearn_svm_classifier(X, y, probability=True):
    clf = svm.SVC(gamma=0.001, C=100., probability=probability, random_state=777)
    model = clf.fit(X, y)
    return model


def create_cuml_svm_classifier(X, y):
    try:
        import cuml
    except ImportError:
        import warnings
        warnings.warn(
            "cuML is required to use GPU explainers. Check https://rapids.ai/start.html for \
            more information on how to install it.",
            ImportWarning)

    clf = cuml.svm.SVC(gamma=0.001, C=100., probability=True)
    model = clf.fit(X, y)
    return model


def create_pandas_only_svm_classifier(X, y, probability=True):
    class PandasOnlyEstimator(TransformerMixin):
        def fit(self, X, y=None, **fitparams):
            return self

        def transform(self, X, **transformparams):
            dataset_is_df = isinstance(X, pd.DataFrame)
            if not dataset_is_df:
                raise Exception("Dataset must be a pandas dataframe!")
            return X

    pandas_only = PandasOnlyEstimator()

    clf = svm.SVC(gamma=0.001, C=100., probability=probability, random_state=777)
    pipeline = Pipeline([('pandas_only', pandas_only), ('clf', clf)])
    return pipeline.fit(X, y)


def create_sklearn_random_forest_regressor(X, y):
    rfr = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=777)
    model = rfr.fit(X, y)
    return model


def wrap_classifier_without_proba(classifier):
    # Wraps a classifier without a predict_proba
    class WrappedWithoutProbaClassificationModel(object):
        """A class for wrapping a classification model."""

        def __init__(self, classifier):
            """Initialize the WrappedWithoutProbaClassificationModel with the underlying model."""
            self._model = classifier

        def predict(self, dataset):
            """Returns the probabilities instead of the predicted class.

            :param dataset: The dataset to predict on.
            :type dataset: DatasetWrapper
            """
            return self._model.predict_proba(dataset)

    return WrappedWithoutProbaClassificationModel(classifier)


def create_sklearn_linear_regressor(X, y, pipeline=False):
    lin = linear_model.LinearRegression()
    if pipeline:
        lin = Pipeline([('lin', lin)])
    model = lin.fit(X, y)
    return model


def create_sklearn_logistic_regressor(X, y, pipeline=False):
    lin = linear_model.LogisticRegression(solver='liblinear')
    if pipeline:
        lin = Pipeline([('lin', lin)])
    model = lin.fit(X, y)
    return model


def create_keras_regressor(X, y):
    # create simple (dummy) Keras DNN model for regression
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('linear'))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model


def create_scikit_keras_model_func(feature_number, num_classes=None):
    def common_scikit_keras_model():
        model = Sequential()
        model.add(Dense(12, input_dim=feature_number, activation='relu'))
        model.add(Dense(8, activation='relu'))
        if num_classes is None:
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        elif num_classes == 2:
            model.add(Dense(2))
            model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        else:
            model.add(Dense(num_classes, activation=Activation('softmax')))
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        return model
    return common_scikit_keras_model


def create_scikit_keras_regressor(X, y):
    # create simple (dummy) Keras DNN model for regression
    batch_size = 500
    epochs = 10
    model_func = create_scikit_keras_model_func(X.shape[1])
    model = KerasRegressor(build_fn=model_func, nb_epoch=epochs, batch_size=batch_size, verbose=1)
    model.fit(X, y)
    return model


def create_scikit_keras_binary_classifier(X, y):
    # create simple (dummy) Keras DNN model for classification
    batch_size = 500
    epochs = 10
    model_func = create_scikit_keras_model_func(X.shape[1], num_classes=2)
    model = KerasClassifier(build_fn=model_func, nb_epoch=epochs, batch_size=batch_size, verbose=1)
    model.fit(X, y)
    return model


def create_scikit_keras_multiclass_classifier(X, y):
    # create simple (dummy) Keras DNN model for classification
    batch_size = 500
    epochs = 10
    num_classes = len(np.unique(y))
    model_func = create_scikit_keras_model_func(X.shape[1], num_classes=num_classes)
    model = KerasClassifier(build_fn=model_func, nb_epoch=epochs, batch_size=batch_size, verbose=1)
    model.fit(X, y)
    return model


def buggy_auc(y_score, y_true):
    # buggy AUC metric which throws when single row is passed
    y_score = y_score[:, 1].flatten()
    y_true = y_true.flatten()
    if y_true.shape[0] == 1:
        raise ValueError('Only one row passed, this is a buggy function!')
    return roc_auc_score(y_true, y_score)


class BuggyCallback(Callback):
    """A buggy fastai callback."""

    def __init__(self):
        """Initialize the BuggyCallback."""
        self.run = True

    def after_epoch(self):
        """Throw an exception after the first epoch."""
        raise ValueError('This buggy callback throws, bummer!')


def _common_fastai_tabular_learner(X, y, is_classifier=True, multimetric=False):
    if is_classifier:
        y_block = CategoryBlock()
    else:
        y_block = RegressionBlock()
    X_copy = X.copy().reset_index(drop=True)
    X_copy[LABEL] = y
    data = TabularDataLoaders.from_df(X_copy, cat_names=[],
                                      cont_names=list(X.columns),
                                      procs=[], y_names=LABEL,
                                      y_block=y_block)
    if multimetric:
        metrics = [accuracy, buggy_auc]
    else:
        metrics = accuracy
    learn = tabular_learner(data, layers=[200, 100], metrics=metrics)
    learn.fit(1, 1e-2)
    if multimetric:
        learn.add_cb(BuggyCallback())
    return learn


def create_fastai_tabular_classifier(X, y):
    return _common_fastai_tabular_learner(
        X, y, is_classifier=True, multimetric=False)


def create_fastai_tabular_classifier_multimetric(X, y):
    return _common_fastai_tabular_learner(
        X, y, is_classifier=True, multimetric=True)


def create_fastai_tabular_regressor(X, y):
    return _common_fastai_tabular_learner(
        X, y, is_classifier=False, multimetric=False)


def _common_pytorch_generator(numCols, num_classes=None):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Apply layer normalization for stability and perf on wide variety of datasets
            # https://arxiv.org/pdf/1607.06450.pdf
            self.norm = nn.LayerNorm(numCols)
            self.fc1 = nn.Linear(numCols, 100)
            self.fc2 = nn.Dropout(p=0.2)
            if num_classes is None:
                self.fc3 = nn.Linear(100, 3)
                self.output = nn.Linear(3, 1)
            elif num_classes == 2:
                self.fc3 = nn.Linear(100, 2)
                self.output = nn.Sigmoid()
            else:
                self.fc3 = nn.Linear(100, num_classes)
                self.output = nn.Softmax()

        def forward(self, X):
            X = self.norm(X)
            X = F.relu(self.fc1(X))
            X = self.fc2(X)
            X = self.fc3(X)
            return self.output(X)
    return Net()


def _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y):
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(torch_X)
        loss = criterion(out, torch_y)
        loss.backward()
        optimizer.step()
        print('epoch: ', epoch, ' loss: ', loss.data.item())
    return net


def create_pytorch_regressor(X, y):
    # create simple (dummy) Pytorch DNN model for regression
    epochs = 12
    if isinstance(X, pd.DataFrame):
        X = X.values
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).float()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1])
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def create_tf_model(inp_ds, val_ds, feature_names):
    """Create a simple TF model for regression.

    :param inp_ds: input data set.
    :type inp_ds: BatchDataset
    :param val_ds: validation data set.
    :type val_ds: BatchDataset
    :param feature_names: list of feature names.
    :type feature_names: list
    :return: a TF model.
    :rtype: tf.keras.Model
    """
    inputs = {col: Input(name=col, shape=(1,),
                         dtype='float32') for col in list(feature_names)}

    x = concatenate([inputs[col] for col in list(feature_names)])
    x = Dense(20, activation='relu', name='hidden1')(x)
    out = Dense(1)(x)

    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', 'mse'])

    model.fit(inp_ds, epochs=5, validation_data=val_ds)
    return model


def create_keras_classifier(X, y):
    # create simple (dummy) Keras DNN model for binary classification
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model


def create_pytorch_classifier(X, y):
    # create simple (dummy) Pytorch DNN model for binary classification
    epochs = 12
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).long()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1], num_classes=2)
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def create_pytorch_single_output_classifier(X, y):
    # create simple (dummy) Pytorch DNN model for binary classification that only outputs
    # the probabilities for the positive class
    epochs = 100
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).float()

    class Net(nn.Module):
        def __init__(self, input_shape):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_shape, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, X):
            X = torch.relu(self.fc1(X))
            X = torch.relu(self.fc2(X))
            return torch.sigmoid(self.fc3(X))

    # Create network structure
    net = Net(X.shape[1])
    # Train the model
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y.reshape(-1, 1))


def create_keras_multiclass_classifier(X, y):
    batch_size = 128
    epochs = 12
    num_classes = len(np.unique(y))
    model = _common_model_generator(X.shape[1], num_classes)
    model.add(Dense(units=num_classes, activation=Activation('softmax')))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    y_train = keras.utils.to_categorical(y, num_classes)
    model.fit(X, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y_train))
    return model


def create_pytorch_multiclass_classifier(X, y):
    # Get unique number of classes
    num_classes = np.unique(y).shape[0]
    # create simple (dummy) Pytorch DNN model for multiclass classification
    epochs = 12
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).long()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1], num_classes=num_classes)
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def create_titanic_pipeline(X_train, y_train):
    def conv(X):
        if isinstance(X, pd.Series):
            return X.values
        return X

    many_to_one_transformer = \
        FunctionTransformer(lambda x: conv(x.sum(axis=1)).reshape(-1, 1))
    many_to_many_transformer = \
        FunctionTransformer(lambda x: np.hstack(
            (conv(np.prod(x, axis=1)).reshape(-1, 1),
                conv(np.prod(x, axis=1)**2).reshape(-1, 1))
        ))
    transformations = ColumnTransformer([
        ("age_fare_1", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ["age", "fare"]),
        ("age_fare_2", many_to_one_transformer, ["age", "fare"]),
        ("age_fare_3", many_to_many_transformer, ["age", "fare"]),
        ("embarked", Pipeline(steps=[
            ("imputer",
                SimpleImputer(strategy='constant', fill_value='missing')),
            ("encoder", OneHotEncoder(sparse=False))]), ["embarked"]),
        ("sex_pclass", OneHotEncoder(sparse=False), ["sex", "pclass"])
    ])
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier',
                           LogisticRegression(solver='lbfgs'))])
    clf.fit(X_train, y_train)
    return clf


def create_dnn_classifier_unfit(feature_number):
    # create simple (dummy) Keras DNN model for binary classification
    model = _common_model_generator(feature_number)
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def create_cancer_data_booleans():
    # Import cancer dataset
    cancer = retrieve_dataset('breast-cancer.train.csv', na_values='?').interpolate().astype('int64')
    cancer_target = cancer.iloc[:, 0]
    cancer_data = cancer.iloc[:, 1:]
    feature_names = cancer_data.columns.values
    target_names = [False, True]
    cancer_target = cancer_target.astype(bool)
    # Split data into train and test
    x_train, x_test, y_train, y_validation = train_test_split(cancer_data, cancer_target,
                                                              test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_validation, feature_names, target_names


def assert_sparse_equal(actual, expected):
    """Assert that two sparse matrices are equal.

    :param actual: The actual sparse matrix.
    :type actual: scipy.sparse.csr_matrix
    :param expected: The expected sparse matrix.
    :type expected: scipy.sparse.csr_matrix
    """
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual.indptr, expected.indptr)
    assert np.array_equal(actual.indices, expected.indices)
    assert np.array_equal(actual.data, expected.data)


def assert_batch_equal(actual, expected):
    """Assert that two batch datasets are equal.

    :param actual: The actual batch dataset.
    :type actual: BatchDataset
    :param expected: The expected batch dataset.
    :type expected: BatchDataset
    """
    zipped_batches = zip(actual, expected)
    for ((c_data, c_label), (data, label)) in zipped_batches:
        assert np.array_equal(c_data, data)
        assert np.array_equal(c_label, label)


def _common_model_generator(feature_number, output_length=1):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(feature_number,)))
    model.add(Dropout(0.25))
    model.add(Dense(output_length, activation='relu', input_shape=(32,)))
    model.add(Dropout(0.5))
    return model
