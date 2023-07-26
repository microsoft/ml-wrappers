.. _model_wrapper_specifications:

Model Wrapper Specifications
============================

In the ML Wrappers SDK, there needs to be a clear understanding of the model type to have a solid contract for users and visualizations. This is particularly important for blackbox models such as those used in the `interpret-community` library. The user can pass in a function from a classifier or regressor, or a model that is a classifier or regressor. For model-specific explainers, the user would pass in the model directly. We can usually infer whether the model is a classifier or regressor in most cases.

Functions
---------

We can evaluate the function on the data and look at the output to understand if the model is a classifier or regressor. In general, if the user passes a function that returns a 1D array, we can infer it is a regressor. If the function returns a 2D array, we can infer it is a classifier. There is a tricky case where the function may return a 2D array of 1 column. In this case, we can throw an exception and force the user to specify model_task=(infer, classifier, regressor), and not allow automatic inferencing. The user can override this behavior if they specify an optional parameter model_task=(infer, classifier, regressor), which will have the value model_task=infer by default.

Models
------

We can convert the model to a function and then use the specifications listed above. We convert the model to a function by either using the predict_proba function, or, if it is not available, the predict function. In some specific cases, we may be able to get additional information from the model to help us decide which function to use. Specifically, if we know that the model is a Keras model, the model will always have a predict_proba method available. In this case, we can look at the shape of predict_proba, and if it has multiple columns or is a single column with values outside the range of [0, 1], we can by default use predict instead. Otherwise, we can use predict_proba. If the user specified model_task=classifier, this will always override the behavior for Keras models and specify whether to use predict or predict_proba. Also, if the user specifies that model_task=classifier, but the model does not have a predict_proba function, we can wrap the function in a one-hot vector of probabilities. After the model is converted to a function that conforms to our specifications, we can wrap that in our model wrapper, which can contain a reference to the original model in cases where it may be needed or for debugging.

Supported Frameworks
--------------------

Our library can directly support the most popular machine learning frameworks. In general, based on the description above, the library can support models and functions in scikit-learn. However, we can extend support to other frameworks with the model wrapper concept. Currently, the list of supported frameworks, or frameworks we plan to support, are:

- Scikit-Learn
- LightGBM
- XGBoost
- Catboost
- Keras with Tensorflow backend
- Pytorch
- ONNX

We would like to support caffe/caffe2 and other ML frameworks in the future as well.