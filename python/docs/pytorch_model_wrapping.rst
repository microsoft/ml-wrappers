.. _pytorch_model_wrapping:

Pytorch Model Wrapping
=======================

The ML Wrappers library provides support for wrapping Pytorch models. This is achieved through the use of model wrappers and utilities specifically designed for Pytorch models.

.. code-block:: python

    import logging
    import numpy as np
    import pandas as pd

    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.INFO)

    try:
        import torch
    except ImportError:
        module_logger.debug('Could not import torch, required if using a PyTorch model')

    try:
        from torchvision.transforms import ToTensor
    except ImportError:
        module_logger.debug('Could not import torchvision, required if using' +
                            ' a vision PyTorch model')

The library attempts to import the necessary Pytorch and torchvision modules. If these imports fail, a debug message is logged indicating that these modules are required when using a Pytorch model.

The library provides a WrappedPytorchModel class for wrapping Pytorch models. This class is used in the wrap_model function to wrap the model if it is a Pytorch model.

.. code-block:: python

    class WrappedPytorchModel(object):
        def __init__(self, model):
            self._model = model

        def predict(self, dataset):
            return self._model(dataset)

        def predict_proba(self, dataset):
            return self._model(dataset)

The WrappedPytorchModel class provides a predict and predict_proba method, which call the model's predict method on the given dataset.

The library also provides a PytorchModelInitializer class for initializing Pytorch models. This class is used in the wrapped_pytorch_model_initializer function to initialize the model.

.. code-block:: python

    class PytorchModelInitializer():
        def __init__(self, model_initializer, model_task):
            self._model_initializer = model_initializer
            self._model_task = model_task

        def __call__(self, X_train, y_train):
            fitted_model = self._model_initializer(X_train, y_train)
            wrapped_pytorch_model = WrappedPytorchModel(fitted_model)
            validate_wrapped_pytorch_model(wrapped_pytorch_model, X_train,
                                           self._model_task)
            return wrapped_pytorch_model

The PytorchModelInitializer class provides a __call__ method, which initializes the model and wraps it using the WrappedPytorchModel class.

.. note::

    The ML Wrappers library only supports Pytorch machine learning models.