.. _object_detection_model_wrapping:

Object Detection Model Wrapping
===============================

ML-Wrappers supports model wrapping of Pytorch object detection methods. The model is converted to a function by either using the predict_proba function, or, if it is not available, the predict function. 

Schema
------
For each image in the dataset, the model is used to generate predictions. Then, the predictions are filtered using non maximal suppression (based on the iuo threshold parameter). 

The predictions is a list of Pytorch tensors. Each tensor is composed of the labels, boxes (bounding boxes), scores. 

Example:

.. code-block:: python

    detections = [{'boxes': tensor([[ 97.0986, 170.7908, 241.4255, 516.5880]], grad_fn=<StackBackward0>), 'labels': tensor([2]), 'scores': tensor([0.9905], grad_fn=<IndexBackward0>)}]

    predict_output = [[[2.0, 97.09860229492188, 170.7908172607422, 241.425537109375, 516.5879516601562, 0.9904877543449402]]]

Limitations
-----------
This wrapper functionality only supports Pytorch machine learning models.

Model Wrapping
--------------
The model wrapping process involves the following steps:

1. Processing the raw detections to generate bounding boxes, class scores, and objectness scores.
2. Applying non-maximal suppression and score filtering based on the iou threshold and score threshold parameters.
3. Creating a list of detection records from the image predictions.

Example:

.. code-block:: python

    class WrappedObjectDetectionModel:
        """A class for wrapping a object detection model in the scikit-learn style."""

        def __init__(self, model: Any, number_of_classes: int, device=Device.AUTO.value) -> None:
            """Initialize the WrappedObjectDetectionModel with the model and evaluation function."""
            self._device = torch.device(_get_device(device))
            model.eval()
            model.to(self._device)

            self._model = model
            self._number_of_classes = number_of_classes

        def predict(self, x, iou_threshold: float = 0.5, score_threshold: float = 0.5):
            """Create a list of detection records from the image predictions."""
            detections = []
            for image in x:
                if type(image) == Tensor:
                    raw_detections = self._model(image.to(self._device).unsqueeze(0))
                else:
                    raw_detections = self._model(T.ToTensor()(image).to(self._device).unsqueeze(0))

Supported Frameworks
--------------------
The following machine learning frameworks are supported:

- Scikit-Learn
- LightGBM
- XGBoost
- Catboost
- Keras with Tensorflow backend
- Pytorch

ONNX is not yet supported, but there are plans to support it in the future. Other ML frameworks like caffe/caffe2 are also planned to be supported in the future.