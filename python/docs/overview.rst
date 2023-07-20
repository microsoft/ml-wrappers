.. _overview:

Overview
========

The ml-wrappers project is a Python library that provides a unified interface for wrapping machine learning models and datasets. It is designed to make it easier to work with different types of models and datasets, and to facilitate the process of explaining and interpreting machine learning models.

The library includes support for a variety of machine learning frameworks, including Scikit-Learn, LightGBM, XGBoost, Catboost, Keras with Tensorflow backend, Pytorch, and ONNX. It also provides a mechanism for inferring whether a model is a classifier or regressor, and for wrapping models in a way that conforms to the specifications required by the library.

The ml-wrappers library also provides a DatasetWrapper class that makes it easier to perform operations such as summarizing data, taking subsets of data, and sampling data. This class can handle a variety of data types, including numpy arrays, pandas DataFrames, pandas Series, scipy sparse matrices, and more.

In addition to wrapping models and datasets, the ml-wrappers library also provides a number of utilities for working with machine learning models. These include functions for evaluating models, generating augmented data, and more.

The library is released under the MIT License and adheres to the Microsoft Open Source Code of Conduct. It is maintained by Microsoft and contributions are welcome.