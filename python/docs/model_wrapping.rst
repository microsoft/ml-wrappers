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

ONNX Runtime
------------

Install ``onnxruntime>=1.19`` separately and create an ``InferenceSession``
with your desired execution providers. Pass the session, not a file path or
an ``onnx.ModelProto``, to ``wrap_model``:

.. code-block:: python

    import onnxruntime as ort
    from ml_wrappers import wrap_model

    session = ort.InferenceSession(
        "model.onnx", providers=["CPUExecutionProvider"])
    wrapped = wrap_model(session, examples, model_task="classification")
    probabilities = wrapped.predict_proba(examples)
    labels = wrapped.predict(examples)

The adapter supports tabular sessions with one rank-two dense tensor input.
Supply NumPy arrays or pandas DataFrames in the exported feature order;
columns are not matched by name. Inputs are cast to the graph's tensor dtype,
so prepare integer/string features appropriately before wrapping. A single
row is accepted as a one-dimensional array. Export a dynamic batch dimension
to predict arbitrary batch sizes. The session's providers are preserved;
the ``device`` option does not configure ONNX Runtime.

Supported outputs are:

* One floating-point vector or single-column matrix for single-target
  regression. Use ``model_task="regression"`` for a single column; the public
  prediction result is flattened.
* One floating-point probability matrix for classification, with a column
  per class (two columns for binary classification). Labels are zero-based
  column indices. No softmax or calibration is added.
* An exported label vector plus either a probability matrix or an ONNX-ML
  ZipMap sequence of class-to-probability dictionaries, as produced by
  ``skl2onnx`` classifiers. Original numeric or string labels are preserved.
  ZipMap probability columns are ordered by sorted class keys; tensor
  columns retain the exported order. A standalone ZipMap output is also
  accepted and its keys provide class labels.

ONNX classifiers must expose at least two probability columns/classes,
including ZipMap outputs. Single-key ZipMaps are rejected rather than
allowing the generic evaluator to expand them into an invented second class.

Multi-input graphs, sparse inputs, label-only classifiers, single-column
binary probabilities, single-class classifiers, multi-target regression, arbitrary auxiliary outputs,
and text/vision ONNX adapters are not supported. Such graphs require an
explicit application-specific adapter or re-export to a supported schema.
Wrapping runs inference on representative examples and can therefore expose
runtime/provider errors immediately.

Text and vision
---------------

Specify the task explicitly for text and image adapters. Supported paths,
optional dependencies and restrictions are listed in
:ref:`supported_frameworks`, :ref:`text_model_wrapping` and
:ref:`image_model_wrapping`. A task constant alone does not guarantee that
every model or pipeline is supported.
