.. _tensorflow_model_wrapping:

TensorFlow and Keras Model Wrapping
===================================

Install a compatible TensorFlow/Keras stack separately.
``wrap_model`` recognizes supported Keras Sequential class namespaces and
adds ``WrappedTensorflowModel`` before evaluating the output contract.
Do not assume automatic recognition of every Keras model subclass.

This complete, untrained demonstration uses a two-column softmax output;
real applications should supply a fitted model:

.. code-block:: python

    import numpy as np
    from tensorflow import keras
    from ml_wrappers import wrap_model

    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(2, activation='softmax'),
    ])
    features = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    wrapped = wrap_model(model, features, model_task='classification')
    probabilities = wrapped.predict_proba(features)
    labels = wrapped.predict(features)

Native Keras models need not implement ``predict_proba`` or
``predict_classes``. The adapter supplies those methods using prediction
output. Its ``predict_proba`` is an alias for ``predict``, not a probability
calibration step. Choose a suitable output layer for classification.

The explicit ``WrappedTensorflowModel`` class is also available for models
with a compatible ``predict`` contract. Use :ref:`model_wrapping` for task
selection and normalized outputs; see :ref:`API Reference` for signatures.
The test suite's initializers and validation helpers are not public APIs.

TensorFlow and SciKeras are excluded from the repository's Python 3.14
development requirements. The base package's tested Python range is not
a promise that every optional framework supports every version.
