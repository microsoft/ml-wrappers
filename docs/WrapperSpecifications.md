# Wrapper Specifications: How to infer classification or regression model type

In the ML Wrappers SDK there needs to be a clear understanding of the model type to have a solid contract for users and visualizations.

For example, in the machine learning interpretability space for blackbox models such as in the https://github.com/interpretml/interpret-community/ library, this means that the user can pass in a function from a classifier or regressor, or a model that is a classifier or regressor. For model-specific explainers, the user would pass in the model directly. We can usually infer whether the model is a classifier or regressor in most cases.

- Functions - We can evaluate the function on the data and look at the output to understand if the model is a classifier or regressor. In general, if the user passes a function that returns a 1D array, we can infer it is a regressor. If the function returns a 2D array, we can infer it is a classifier. There is a tricky case where the function may return a 2D array of 1 column. In this case, we can throw an exception and force the user to specify model_task=(infer, classifier, regressor), and not allow automatic inferencing. The user can override this behavior if they specify an optional parameter model_task=(infer, classifier, regressor), which will have the value model_task=infer by default.

  - If they specify model_task=infer:
    - We will try to infer whether the function is for classification or regression based on the specifications above.
  - If they specify model_task=classifier and:
    - They have a 2D array - run function, treat output as classifier
    - They have a 1D array - add wrapper function to convert output to 2D array. Run function on samples and assert all values are probabilities. If are not all 1, convert to a 2D array with 2 columns [1-p, p]. If they are greater than 1, throw exception.
    - They pass in classes parameter - run function, treat output as classifier
  - If they have model_task=regressor and:
    - They have a 2D array - if it has 1 column, treat it as regressor, if more than one column throw exception
    - They have a 1D array - run function, treat output as regressor
    - They pass in a classes parameter - throw exception, since user specified they are not using a classifier

Note for some types of frameworks, like catboost, we have found that the prediction results (in this case the predicted probabilities) for a single instance may be of a different shape than prediction results for multiple instances.  In this scenario, we can call the model for both single and multiple instances and compare the output dimensionality, and if they differ by one, wrap the prediction function to add an additional dimension if a single instance is predicted on.

- Models - We can convert the model to a function and then use the specifications listed above. We convert the model to a function by either using the predict_proba function, or, if it is not available, the predict function. In some specific cases, we may be able to get additional information from the model to help us decide which function to use. Specifically, if we know that the model is a Keras model, the model will always have a predict_proba method available. In this case, we can look at the shape of predict_proba, and if it has multiple columns or is a single column with values outside the range of [0, 1], we can by default use predict instead. Otherwise, we can use predict_proba. If the user specified model_task=classifier, this will always override the behavior for Keras models and specify whether to use predict or predict_proba. Also, if the user specifies that model_task=classifier, but the model does not have a predict_proba function, we can wrap the function in a one-hot vector of probabilities. After the model is converted to a function that conforms to our specifications, we can wrap that in our model wrapper, which can contain a reference to the original model in cases where it may be needed or for debugging.

- Supported Frameworks - Our library can directly support the most popular machine learning frameworks. In general, based on the description above, the library can support models and functions in scikit-learn. However, we can extend support to other frameworks with the model wrapper concept. Currently, the list of supported frameworks, or frameworks we plan to support, are:
  - Scikit-Learn - This framework is directly supported by our APIs.
  - LightGBM - We can wrap the function into a scikit-learn compatible wrapper.
  - XGBoost - We can wrap the function into a scikit-learn compatible wrapper.
  - Catboost - We can wrap the function into a scikit-learn compatible wrapper.
  - Keras with Tensorflow backend - Keras has both a predict_proba and predict function on all models, so it is difficult to know for sure if the model is a classifier or regressor. We can force the user to specify whether the model is a classifier or regressor in case only a single column is output, and then wrap the model in a model wrapper. If the user specifies the model is a regressor we can fix the structure to be 2D.
  - Pytorch - Pytorch does not have a predict or predict_proba function, but the model can be called on the dataset directly to get probabilities. The probabilities can then be transformed into predicted labels for classifiers. Similarly to Keras, we can force the user to specify whether the model is a classifier or regressor in case only a single column is output, and then wrap the model in a model wrapper. If the user specifies the model is a regressor we can fix the structure to be 2D.
  - ONNX - ONNX is not yet supported, but we plan to support it in the future. We can use a model wrapper to conform to the predict and predict_proba specifications the SDK requires.

We would like to support caffe/caffe2 and other ML frameworks in the future as well.  Please feel free to contribute to this repository.