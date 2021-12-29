# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

class DatasetConstants(object):
    """Dataset related constants."""
    CATEGORICAL = 'categorical'
    CLASSES = 'classes'
    FEATURES = 'features'
    NUMERIC = 'numeric'
    X_TEST = 'x_test'
    X_TRAIN = 'x_train'
    Y_TEST = 'y_test'
    Y_TRAIN = 'y_train'


class ModelType(object):
    """Model type constants."""
    XGBOOST = 'xgboost'
    TREE = 'tree'
    DEFAULT = 'default'
