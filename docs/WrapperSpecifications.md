# Wrapper Specifications: How to infer classification or regression model type

In the ML Wrappers SDK there needs to be a clear understanding of the model type to have a solid contract for users and visualizations.

The detailed rules below describe the intended model and function contract.
They remain the repository's design specification, including requirements
that are not yet fully implemented. Current implementation differences are
called out separately at the end, rather than removing those requirements.
Task names use the current API values: `'unknown'`, `'classification'` and
`'regression'` (previously described here as infer, classifier and regressor).

For example, in the machine learning interpretability space for blackbox models such as in the https://github.com/interpretml/interpret-community/ library, this means that the user can pass in a function from a classifier or regressor, or a model that is a classifier or regressor. For model-specific explainers, the user would pass in the model directly. We can usually infer whether the model is a classifier or regressor in most cases.

## Functions

We can evaluate the function on the data and look at the output to understand if the model is a classifier or regressor. In general, if the user passes a function that returns a 1D array, we can infer it is a regressor. If the function returns a 2D array, we can infer it is a classifier. There is a tricky case where the function may return a 2D array of 1 column. In this case, we can throw an exception and force the user to specify `model_task='classification'` or `model_task='regression'`, and not allow automatic inferencing. The user can override automatic inference with the optional `model_task` parameter, which defaults to `'unknown'`.

- If they specify `model_task='unknown'`:
  - We will try to infer whether the function is for classification or regression based on the specifications above.
- If they specify `model_task='classification'` and:
  - They have a 2D array - run the function and treat the output as a classifier.
  - They have a 1D array - add a wrapper function to convert the output to a 2D array. Run the function on samples and assert all values are probabilities in [0, 1]. If they are not all 1, convert to a 2D array with 2 columns `[1-p, p]`. If all values are 1, retain a single probability column for the one-class case. If values are outside [0, 1], throw an exception.
  - They pass in a `classes` parameter - run the function and treat the output as a classifier.
- If they specify `model_task='regression'` and:
  - They have a 2D array - if it has 1 column, treat it as a regressor; if it has more than one column, throw an exception.
  - They have a 1D array - run the function and treat the output as a regressor.
  - They pass in a `classes` parameter - throw an exception, since the user specified they are not using a classifier.

Note for some types of frameworks, like CatBoost, we have found that the prediction results (in this case the predicted probabilities) for a single instance may be of a different shape than prediction results for multiple instances. In this scenario, we can call the model for both single and multiple instances and compare the output dimensionality, and if they differ by one, wrap the prediction function to add an additional dimension if a single instance is predicted on.

## Models

We can convert the model to a function and then use the specifications listed above. We convert the model to a function by either using the `predict_proba` function, or, if it is not available, the `predict` function. In some specific cases, we may be able to get additional information from the model to help us decide which function to use.

For Keras, a framework adapter can expose both prediction methods even when the native model does not provide `predict_proba`. The intended selection rule is to look at the shape of `predict_proba`: if it has multiple columns or is a single column with values outside [0, 1], use `predict` by default; otherwise use `predict_proba`. An explicit classification or regression task should override automatic selection. This design rule should not be mistaken for the current evaluator's behavior; see the implementation notes below.

If the user specifies `model_task='classification'`, but the model does not have a `predict_proba` function, we can wrap the function in a one-hot vector of probabilities. These are class indicators, not calibrated probabilities. After the model is converted to a function that conforms to our specifications, we can wrap that in our model wrapper, which can contain a reference to the original model in cases where it may be needed or for debugging.

## Supported and planned frameworks

Our library can directly support the most popular machine learning frameworks. In general, based on the description above, the design can support models and functions in scikit-learn. We can extend support to other frameworks with the model wrapper concept. The original framework-specific requirements and planned support are:

- Scikit-Learn - This framework is directly supported by our APIs.
- LightGBM - We can wrap the function into a scikit-learn compatible wrapper.
- XGBoost - We can wrap the function into a scikit-learn compatible wrapper.
- CatBoost - We can wrap the function into a scikit-learn compatible wrapper.
- Keras with TensorFlow backend - A wrapper can provide both `predict_proba` and `predict`, so the presence of those methods alone does not identify a classifier or regressor. We can force the user to specify the task when only a single column is output, and then wrap the model. The original design calls for normalizing regression output to 2D; the current implementation instead normalizes it to 1D, as noted below.
- PyTorch - PyTorch modules do not normally have a `predict` or `predict_proba` function, but the model can be called on the dataset directly. For classification the model must supply probabilities, which can then be transformed into predicted labels; the wrapper does not add softmax automatically. Similarly to Keras, we can require an explicit task when only a single column is output, and then wrap the model. The original design calls for normalizing regression output to 2D; the current implementation instead normalizes it to 1D.
- ONNX - ONNX Runtime sessions now receive an adapter conforming their output to the `predict` and `predict_proba` specifications. The implemented tabular contract supports one dense rank-two input and a floating-point prediction tensor, or exported labels plus tensor/ZipMap class probabilities. Original exported labels are preserved; probability-only tensors use column indices. ONNX classifiers currently require at least two probability columns/classes, including ZipMap outputs; single-column binary probabilities require re-export as two columns. Multi-input graphs and text/vision ONNX adapters remain outside this contract. See the [usage and output schemas](https://ml-wrappers.readthedocs.io/en/latest/model_wrapping.html#onnx-runtime).

Support for Caffe/Caffe2 and MXNet was also identified as future work. Those integrations are now not planned: MXNet is retired and Caffe/Caffe2 are legacy ecosystems (see closed issues [#15](https://github.com/microsoft/ml-wrappers/issues/15) and [#8](https://github.com/microsoft/ml-wrappers/issues/8)). Contributions for other actively maintained frameworks are welcome.

## Current implementation differences

The specification above preserves the intended behavior. For applications using
the current package, the following distinctions matter:

- Public `wrap_model` takes a supported model and returns one model, not an internal `(model, task)` tuple. Bare tabular prediction callables are not generally accepted by that public entry point; function evaluation is handled by internal helpers.
- `classes` is currently forwarded to the image path. The general tabular `classes` rules above are design requirements, not implemented classification overrides or regression validation.
- The ordinary evaluator prefers `predict_proba` when present unless regression is explicitly requested. It does not implement the Keras range/shape-based method-selection heuristic above.
- Automatic one-hot adaptation of a classifier without `predict_proba` currently applies to `SGDClassifier`, not every classifier.
- Single-target regression output is currently flattened to 1D, including an original single-column output. Multi-column regression is rejected. Automatic inference also has a special case treating a one-column pandas DataFrame output as regression.

See the [current public API contract](https://ml-wrappers.readthedocs.io/en/latest/model_wrapper_specifications.html)
and [scenario support matrix](https://ml-wrappers.readthedocs.io/en/latest/supported_frameworks.html)
for usage, and [model_wrapper.py](../python/ml_wrappers/model/model_wrapper.py),
[evaluator.py](../python/ml_wrappers/model/evaluator.py) and
[function_wrapper.py](../python/ml_wrappers/model/function_wrapper.py)
for the implementation of dispatch, inference and shape normalization.
