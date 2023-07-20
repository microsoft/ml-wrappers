.. _supported_models:

Supported Models
================

The ML-Wrappers library supports a variety of machine learning models. The following sections provide an overview of the supported models.

Scikit-Learn
------------

Scikit-Learn models are directly supported by our APIs. 

LightGBM
--------

LightGBM models can be wrapped into a scikit-learn compatible wrapper.

XGBoost
-------

XGBoost models can be wrapped into a scikit-learn compatible wrapper.

Catboost
--------

Catboost models can be wrapped into a scikit-learn compatible wrapper.

Keras with Tensorflow backend
-----------------------------

Keras models have both a predict_proba and predict function on all models, so it is difficult to know for sure if the model is a classifier or regressor. We can force the user to specify whether the model is a classifier or regressor in case only a single column is output, and then wrap the model in a model wrapper. If the user specifies the model is a regressor we can fix the structure to be 2D.

Pytorch
-------

Pytorch does not have a predict or predict_proba function, but the model can be called on the dataset directly to get probabilities. The probabilities can then be transformed into predicted labels for classifiers. Similarly to Keras, we can force the user to specify whether the model is a classifier or regressor in case only a single column is output, and then wrap the model in a model wrapper. If the user specifies the model is a regressor we can fix the structure to be 2D.

ONNX
----

ONNX is not yet supported, but we plan to support it in the future. We can use a model wrapper to conform to the predict and predict_proba specifications the SDK requires.

Future Support
--------------

We would like to support caffe/caffe2 and other ML frameworks in the future as well. Contributions to this repository are welcome.