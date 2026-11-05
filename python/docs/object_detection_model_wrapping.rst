.. _object_detection_model_wrapping:

Object Detection Model Wrapping
===============================

Detection adapters return records per image rather than a tabular
class-probability matrix. They require PyTorch/torchvision; the AzureML
AutoML Images MLflow path also requires its framework dependencies.

Torchvision-style detector
--------------------------

Given a fitted ``detector`` with ``eval`` and ``to`` methods, whose inference
returns a list of dictionaries containing ``boxes``, ``labels`` and
``scores`` tensors:

.. code-block:: python

    from ml_wrappers import wrap_model

    wrapped = wrap_model(
        detector, images, model_task='object_detection',
        num_classes=3, device='cpu')
    detections = wrapped.predict(images)
    scores = wrapped.predict_proba(images)

Here ``images`` is an iterable of image tensors or images compatible with
torchvision ``ToTensor``. The adapter invokes the detector one image at a
time. It is not a universal adapter for arbitrary Transformers detection
pipeline dictionaries. The public dispatch automatically adapts compatible
callable detectors without ``predict`` or ``predict_proba`` methods, including
torchvision ``torch.nn.Module`` detectors.

Alternatively, construct the same adapter explicitly:

.. code-block:: python

    from ml_wrappers.model.image_model_wrapper import WrappedObjectDetectionModel

    wrapped = WrappedObjectDetectionModel(
        detector, number_of_classes=3, device='cpu')

.. important::

   In the current ``WrappedObjectDetectionModel``, ``predict`` accepts
   ``iou_threshold`` and ``score_threshold`` but does not honor overrides.
   NMS and confidence filtering use fixed thresholds of 0.5.
   ``predict_proba`` also uses that fixed filtering. This limitation does
   not apply to the AutoML MLflow adapter's ``predict`` threshold parameters.

Record schema
--------------

Each detection record is ``[label, x_min, y_min, x_max, y_max, score]``.
Coordinates are corners, not width and height; the confidence score is
included in wrapped predictions. For example::

    [
        [[1, 10, 20, 50, 80, 0.9], [2, 30, 40, 70, 90, 0.8]],
        [[1, 15, 25, 55, 85, 0.95]],
    ]

The outer sequence corresponds to images. Equal detection counts can produce
a NumPy array; differing counts produce nested lists. Handle ragged results
instead of assuming a rectangular array. ``predict_proba`` returns the score
for each retained detection, not a complete class distribution.

AutoML Images MLflow
--------------------

Use a recognized AutoML Images MLflow model with ordered ``classes``:

.. code-block:: python

    from ml_wrappers import wrap_model

    wrapped = wrap_model(
        automl_model, examples, model_task='object_detection',
        classes=['cat', 'dog'])
    detections = wrapped.predict(
        examples, iou_threshold=0.4, score_threshold=0.6)

``examples`` must be the AutoML image input DataFrame plus an ``image_size``
column containing each image's ``(width, height)``. The wrapper removes that
column before model inference and rescales normalized boxes. If every label
in an image is integer-convertible, those integer values are preserved.
Otherwise all labels in that image are mapped through ``classes`` to
one-based indices in the supplied order. This adapter honors the two
``predict`` thresholds.

``PytorchDRiseWrapper`` and ``MLflowDRiseWrapper`` expose separate D-RISE
contracts; they are not interchangeable with these detection-record outputs.
See :ref:`API Reference` for their signatures and return values.
