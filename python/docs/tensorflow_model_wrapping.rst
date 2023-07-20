.. _tensorflow_model_wrapping:

Tensorflow Model Wrapping
=========================

The ML Wrappers library provides support for wrapping Tensorflow models to conform to the required specifications for model explanations. This is achieved through the ``WrappedTensorflowModel`` class and the ``is_sequential`` function.

WrappedTensorflowModel
----------------------

The ``WrappedTensorflowModel`` class is used to wrap a Tensorflow model. This class is initialized with the model to be wrapped. It provides the ``predict`` method for making predictions using the wrapped Tensorflow model.

.. code-block:: python

    class WrappedTensorflowModel(object):
        def __init__(self, model):
            self._model = model

        def predict(self, dataset):
            if isinstance(dataset, pd.DataFrame):
                dataset = dataset.values
            return self._model.predict(dataset)

is_sequential
-------------

The ``is_sequential`` function checks if a given model is a sequential model. It returns True if the model is a sequential model and False otherwise.

.. code-block:: python

    def is_sequential(model):
        return str(type(model)).endswith("keras.engine.sequential.Sequential'>")

Tensorflow Model Initializer
----------------------------

The Tensorflow Model Initializer is a class that initializes a Tensorflow model and wraps it using the ``WrappedTensorflowModel`` class. It also validates the wrapped Tensorflow model.

.. code-block:: python

    class TensorflowModelInitializer():
        def __init__(self, model_initializer, model_task):
            self._model_initializer = model_initializer
            self._model_task = model_task

        def __call__(self, X_train, y_train):
            fitted_model = self._model_initializer(X_train, y_train)
            wrapped_tf_model = WrappedTensorflowModel(fitted_model)
            validate_wrapped_tf_model(wrapped_tf_model, X_train, self._model_task)
            return wrapped_tf_model

The ``wrapped_tensorflow_model_initializer`` function returns an instance of the TensorflowModelInitializer class.

.. code-block:: python

    def wrapped_tensorflow_model_initializer(model_initializer, model_task):
        return TensorflowModelInitializer(model_initializer, model_task)

Supported Frameworks
--------------------

The ML Wrappers library supports a variety of machine learning frameworks. For Tensorflow models, the library can wrap the model in a model wrapper if the user specifies whether the model is a classifier or regressor in case only a single column is output. If the user specifies the model is a regressor, the structure can be fixed to be 2D.

.. note::

    The library can directly support the most popular machine learning frameworks. However, support can be extended to other frameworks with the model wrapper concept.