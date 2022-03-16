# Machine Learning Wrappers

## Overview and Motivation
Responsible AI tools should be able to work with a broad spectrum of machine learning models and datasets. Much of this functionality is based on the ability to call predict or predict_proba on a model and get back the predicted values or probabilities in a specific format.

However, there are many different models outside of scikit-learn and even within scikit-learn which have unusual outputs or require the input in a specific format.  Some, like pytorch, don’t even have the predict/predict_proba function specification.

We initially started adding wrappers in the https://github.com/interpretml/interpret-community repository but found that they are needed by other teams as well, including https://github.com/fairlearn/fairlearn and https://github.com/microsoft/responsible-ai-toolbox, hence the code has been moved to this repository.  Anyone is welcome to use or contribute to these model and dataset wrappers.

These wrappers handle a variety of frameworks, including pytorch, tensorflow, keras wrappers on tensorflow, variations on scikit-learn models (such as the SVC classification model that doesn’t have a predict_proba function), lightgbm and xgboost, as well as certain strange pipelines we have encountered from customers and internal users in the past.

The dataset wrapper handles a variety of different dataset types and converts them to a common numpy or scipy sparse format for internal code to handle in one simple way.  Hence, the code doesn’t have to worry about whether the current input is pandas or some other format, it doesn’t have to include if/else branches everywhere in the code.

The dataset wrapper simply converts the input to the common format, and after the common code finishes running, we convert the representation back to the original format, which can be handled by the original model.

Currently supported data types include:

- numpy.ndarray
- pandas.DataFrame
- panads.Series
- scipy.sparse.csr_matrix
- shap.DenseData
- torch.Tensor
- tensorflow.python.data.ops.dataset_ops.BatchDataset

For more information about common format from the wrappers, please see the [Wrapper Specifications](https://github.com/microsoft/ml-wrappers/tree/main/docs/WrapperSpecifications.md) documentation.

## Installation

To install the package, simply run:

```
pip install ml-wrappers
```

## Code example of wrap_model

```python
from ml_wrappers import wrap_model
wrapped_model = wrap_model(model, input, model_task='regression')
# Use wrapped model in any common code
```

## Code example of DatasetWrapper

```python
from ml_wrappers import DatasetWrapper
wrapped_dataset = DatasetWrappper(input)
numpy_or_scipy = wrapped_dataset.dataset
# Perform some operations on common converted numpy or scipy dataset
...
# Get back the original dataset type after modifications
modified_input = wrapped_dataset.typed_dataset(numpy_or_scipy)
```

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
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
