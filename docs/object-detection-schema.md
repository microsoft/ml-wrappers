# Object detection schema

The canonical [object detection guide](https://ml-wrappers.readthedocs.io/en/latest/object_detection_model_wrapping.html)
documents adapter selection, dependencies, examples and limitations.
Its [source](../python/docs/object_detection_model_wrapping.rst) is available
for changes not yet published to Read the Docs.

Wrapped detection records have the form
`[label, x_min, y_min, x_max, y_max, score]`, grouped by image.
Unequal detection counts produce nested lists rather than a rectangular array.
`predict_proba` exposes retained detection scores, not a class distribution.

Compatible callable torchvision-style detectors are automatically adapted by
`wrap_model(..., model_task='object_detection')`; constructing
`WrappedObjectDetectionModel` explicitly is an alternative.
AzureML AutoML Images MLflow models have a separate adapter and input contract.
The current torchvision-style adapter uses fixed IoU and confidence thresholds
of 0.5 even when overrides are supplied; the AutoML adapter's `predict`
honors its threshold parameters. D-RISE adapters have different contracts.
