.. _model_wrapping:

Model Wrapping
==============

Import ``wrap_model`` from ``ml_wrappers`` and pass a fitted model plus
representative examples accepted by that model. Wrapping can invoke inference
to inspect shapes; it does not train the model.

.. testcode::

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from ml_wrappers import wrap_model

    features = np.array([[0.], [1.], [2.]])
    model = LinearRegression().fit(features, [0., 2., 4.])
    wrapped = wrap_model(model, features, model_task='regression')
    assert wrapped.predict(features).shape == (3,)
    assert np.allclose(wrapped.predict(features), [0., 2., 4.])

Task selection
--------------

Tabular tasks are ``'classification'``, ``'regression'`` and ``'unknown'``
(the default), also available as ``ModelTask`` constants from
``ml_wrappers.common.constants``. Do not use ``infer``, ``classifier`` or
``regressor`` as task strings.

For ordinary tabular models, automatic selection prefers ``predict_proba``
when available; explicit regression selects ``predict``. Ambiguous
single-column output may require an explicit task. For classification,
provide class probabilities rather than arbitrary scores or class labels.
The adapter does not generally calibrate scores.

PyTorch modules and recognized Keras Sequential models receive framework
adapters before evaluation. These adapters provide ``predict_proba`` as an
alias for their prediction output; this does not mean native Keras models
have that method. Automatic handling of classifiers without ``predict_proba``
is limited to ``SGDClassifier`` and produces one-hot class indicators, not
calibrated probabilities. For example, fit ``SVC(probability=True)`` when
probabilities are needed.

Return value and inputs
-----------------------

The public function returns a single wrapped model. Use ``predict`` for
regression values or classification labels, and ``predict_proba`` for
classification probability output. Single-target regression returns a
one-dimensional array, including when a model originally returns one column.
See :ref:`model_wrapper_specifications` for the detailed contract.

Ordinary tabular models must expose a supported prediction method; a bare
prediction callable is not a generally supported public ``wrap_model`` input.
Use :ref:`dataset_wrapping` to understand pandas reconstruction.

Text and vision
---------------

Specify the task explicitly for text and image adapters. Supported paths,
optional dependencies and restrictions are listed in
:ref:`supported_frameworks`, :ref:`text_model_wrapping` and
:ref:`image_model_wrapping`. A task constant alone does not guarantee that
every model or pipeline is supported.
