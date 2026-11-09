# Machine Learning Wrappers
![PyPI](https://img.shields.io/pypi/v/ml-wrappers)
![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)
[![Tested Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python.yml)
[![Monthly downloads](https://img.shields.io/pypi/dm/ml-wrappers)](https://pypistats.org/packages/ml-wrappers)

[![CI Python minimal environment](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python-minimal.yml/badge.svg)](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python-minimal.yml)
[![CI Python](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python.yml/badge.svg)](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python.yml)
[![CI Python AutoML](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python-AutoML.yml/badge.svg)](https://github.com/microsoft/ml-wrappers/actions/workflows/CI-python-AutoML.yml)


## Overview and Motivation
Responsible AI tools should be able to work with a broad spectrum of machine learning models and datasets. Much of this functionality is based on the ability to call predict or predict_proba on a model and get back the predicted values or probabilities in a specific format.

However, there are many different models outside of scikit-learn and even within scikit-learn which have unusual outputs or require the input in a specific format.  Some, like pytorch, don’t even have the predict/predict_proba function specification.

We initially started adding wrappers in the https://github.com/interpretml/interpret-community repository but found that they are needed by other teams as well, including https://github.com/fairlearn/fairlearn and https://github.com/microsoft/responsible-ai-toolbox, hence the code has been moved to this repository.  Anyone is welcome to use or contribute to these model and dataset wrappers.

These wrappers handle a variety of frameworks, including PyTorch, TensorFlow/Keras, scikit-learn, LightGBM and XGBoost. Supported adapters depend on the task; see the [framework support guide](https://ml-wrappers.readthedocs.io/en/latest/supported_frameworks.html). For SVC classification, enable `probability=True` before fitting; automatic adaptation of classifiers without `predict_proba` is limited to `SGDClassifier`.

The dataset wrapper exposes pandas inputs as NumPy arrays and retains NumPy and SciPy sparse inputs, allowing common operations without repeatedly branching on the original input type. Framework-specific inputs have additional conversion constraints; see the [dataset guide](https://ml-wrappers.readthedocs.io/en/latest/dataset_wrapping.html).

The dataset wrapper simply converts the input to the common format, and after the common code finishes running, we convert the representation back to the original format, which can be handled by the original model.

Currently supported data types include:

- numpy.ndarray
- pandas.DataFrame
- pandas.Series
- scipy.sparse.csr_matrix
- shap.DenseData
- torch.Tensor
- tensorflow.python.data.ops.dataset_ops.BatchDataset

See the [documentation](https://ml-wrappers.readthedocs.io/en/latest/) for runnable quickstarts, [wrapper contracts](https://ml-wrappers.readthedocs.io/en/latest/model_wrapper_specifications.html), framework prerequisites and API reference. `latest` describes `main`; installed releases can differ.

## Installation

To install the package, simply run:

```
pip install ml-wrappers
```

## Code example of wrap_model

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from ml_wrappers import wrap_model

features = np.array([[0.0], [1.0], [2.0]])
model = LinearRegression().fit(features, [0.0, 2.0, 4.0])
wrapped_model = wrap_model(model, features, model_task='regression')
predictions = wrapped_model.predict(features)
assert predictions.shape == (3,)
```

## Code example of DatasetWrapper

```python
import pandas as pd
from ml_wrappers import DatasetWrapper

data = pd.DataFrame({'age': [20, 30], 'income': [100, 200]})
wrapped_dataset = DatasetWrapper(data)
modified_array = wrapped_dataset.dataset.copy()
modified_array[:, 0] += 1
modified_input = wrapped_dataset.typed_wrapper_func(modified_array)
assert isinstance(modified_input, pd.DataFrame)
assert modified_input['age'].tolist() == [21, 31]
```

`typed_dataset` is a property for restoring the wrapper's current data.
Use `typed_wrapper_func(array)` to restore a separately modified array.

<!-- markdown-link-check-disable -->

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

<!-- markdown-link-check-enable -->
