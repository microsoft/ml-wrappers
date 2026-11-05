# Object detection schema

ML Wrappers converts compatible torchvision-style object detector outputs
into per-image detection records. For each image, the wrapper runs inference,
applies non-maximum suppression (NMS) and confidence-score filtering, then
combines the retained labels, bounding boxes and scores.

## Schema and example

The raw model output is a list of dictionaries, one per image. Each dictionary
contains PyTorch tensors:

- `boxes`: bounding boxes as `[x_min, y_min, x_max, y_max]`.
- `labels`: the predicted class ID for each box.
- `scores`: the confidence score for each prediction.

For example, a model might return the following single-image detection
(tensor display values are rounded):

```text
detections = [
    {
        'boxes': tensor([[97.0986, 170.7908, 241.4255, 516.5880]]),
        'labels': tensor([2]),
        'scores': tensor([0.9905]),
    }
]
```

The wrapper combines those fields into records of the form
`[label, x_min, y_min, x_max, y_max, score]`. For the example above, the
wrapped prediction contains one image with one detection, shown here as
nested lists for readability:

```text
predict_output = [
    [
        [2.0, 97.09860229492188, 170.7908172607422,
         241.425537109375, 516.5879516601562, 0.9904877543449402]
    ]
]
```

The coordinates describe box corners, not width and height. The outer
sequence groups detections by image; each image can have multiple records.
Equal detection counts can produce a NumPy array; unequal counts produce
nested lists rather than a rectangular array.
`predict_proba` exposes retained detection scores, not a class distribution.

## Adapters and limitations

Compatible callable torchvision-style detectors are automatically adapted by
`wrap_model(..., model_task='object_detection')`; constructing
`WrappedObjectDetectionModel` explicitly is an alternative.
AzureML AutoML Images MLflow models have a separate adapter and input contract.
The current torchvision-style adapter uses fixed IoU and confidence thresholds
of 0.5 even when overrides are supplied; the AutoML adapter's `predict`
honors its threshold parameters. D-RISE adapters have different contracts.

See the [object detection guide](https://ml-wrappers.readthedocs.io/en/latest/object_detection_model_wrapping.html)
for adapter usage and dependencies, or its
[source](../python/docs/object_detection_model_wrapping.rst) for changes not
yet published to Read the Docs.
