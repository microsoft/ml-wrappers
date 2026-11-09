.. _model_wrapper_specifications:

Model Wrapper Specifications
============================

This page describes the implemented public contract, rather than proposed
future behavior. The detailed
`repository design specification <https://github.com/microsoft/ml-wrappers/blob/main/docs/WrapperSpecifications.md>`_
preserves the intended function/model inference rules and identifies
implementation differences. See :ref:`model_wrapping` for usage and
:ref:`supported_frameworks` for scenario-specific adapters.

Inputs
------

``wrap_model(model, examples, model_task='unknown', num_classes=None,
classes=None, device='auto')`` returns one model. ``examples`` must be valid,
representative inputs because wrapping can run inference to determine output
shape. Tabular examples may be a ``DatasetWrapper`` or a supported dataset
such as a NumPy array, pandas DataFrame or SciPy sparse matrix.

``'unknown'`` requests automatic tabular inference. Use ``'classification'``
or ``'regression'`` to disambiguate output, particularly one-column
predictions. Text and vision require the appropriate explicit task.
``classes``, ``num_classes`` and ``device`` are vision-path options, not a
general mechanism for assigning tabular class labels.

Tabular outputs
---------------

* Classification ``predict`` returns a label per example; ``predict_proba``
  returns a two-dimensional array with one column per class. Binary
  probability output may be expanded to two columns.
* Single-target regression ``predict`` returns a one-dimensional array.
  A multi-column regression output is not supported by this contract.
* The ordinary evaluator uses ``predict_proba`` if present, except when
  regression is explicitly requested; otherwise it uses ``predict``.
* Framework adapters can supply missing methods, but generally do not add
  softmax, calibration or probability validation. The caller must supply
  output suitable for the requested task.
* Automatic no-probability classification adaptation is specific to
  ``SGDClassifier``; do not assume any estimator exposing only ``predict``
  is automatically recognized as a classifier.

Other scenarios
---------------

Text classification, question answering, image classification and detection
have their own input and output contracts. See :ref:`text_model_wrapping`,
:ref:`image_model_wrapping` and :ref:`object_detection_model_wrapping`.
In particular, detection probabilities are per-detection scores, not a
tabular class-probability matrix.

The private ``_wrap_model`` and related evaluator functions may return
internal ``(model, task)`` tuples. Those are not the public ``wrap_model``
return value. For ONNX Runtime session contracts and limitations, see
:ref:`model_wrapping`.
