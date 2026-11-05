.. _image_model_wrapping:

Image Model Wrapping
====================

Pass ``model_task='image_classification'`` to ``wrap_model`` for a supported
image classifier. The public function returns one model, not a tuple.
Image data layout, preprocessing and dependencies are framework-specific.

The dispatch paths include PyTorch modules, compatible FastAI learners,
Transformers/callable classification pipelines and recognized AzureML
AutoML Images MLflow models. An arbitrary callable, MLflow model or image
framework is not necessarily compatible with these contracts.

For a fitted CPU PyTorch classifier ``model`` producing class probabilities,
and an array ``images`` of images accepted by its preprocessing:

.. code-block:: python

    from ml_wrappers import wrap_model

    wrapped = wrap_model(model, images, model_task='image_classification')
    labels = wrapped.predict(images)
    probabilities = wrapped.predict_proba(images)

The PyTorch image adapter uses torchvision tensor conversion for non-tensor
images. Supply the channel order and value range expected by that conversion
and the model; normalization and resizing remain the caller's responsibility.
It does not add softmax. The direct NumPy output conversion requires CPU
outputs. FastAI also has a multilabel image path selected with
``model_task='multilabel_image_classification'``.

Object detection
-----------------

Detection uses a different output schema and adapters. For the AutoML MLflow
path, ``classes`` supplies ordered class names. ``num_classes`` and ``device``
apply to the PyTorch-style detection adapter path; ``device='auto'`` chooses
CUDA when available, otherwise CPU.

Compatible callable detectors without ``predict`` or ``predict_proba``,
including raw torchvision modules, are automatically adapted by the public
detection dispatch. Constructing ``WrappedObjectDetectionModel`` explicitly
is an alternative; see :ref:`object_detection_model_wrapping` for both paths.
