# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines wrappers for vision-based models."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.dataset.dataset_wrapper import DatasetWrapper
from ml_wrappers.model.evaluator import _eval_model
from ml_wrappers.model.pytorch_wrapper import WrappedPytorchModel
from ml_wrappers.model.wrapped_classification_model import \
    WrappedClassificationModel

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

try:
    import torch
    import torch.nn as nn
except ImportError:
    module_logger.debug('Could not import torch, required if using a' +
                        'PyTorch model')

try:
    from vision_explanation_methods.explanations import common as od_common
except ImportError:
    module_logger.debug('Could not import vision_explanation_methods,' +
                        'required if using DRISE')

try:
    import torchvision
    from torchvision import transforms as T
except ImportError:
    module_logger.debug('Could not import torchvision, required if' +
                        'using DRISE')

try:
    from mlflow.pyfunc import PyFuncModel
except ImportError:
    PyFuncModel = Any
    module_logger.debug('Could not import mlflow, required if using an' +
                        'mlflow model')

FASTAI_MODEL_SUFFIX = "fastai.learner.Learner'>"
BOXES = 'boxes'
LABELS = 'labels'
SCORES = 'scores'


def _is_fastai_model(model):
    """Check if the model is a fastai model.

    :param model: The model to check.
    :type model: object
    :return: True if the model is a fastai model, False otherwise.
    :rtype: bool
    """
    return str(type(model)).endswith(FASTAI_MODEL_SUFFIX)


def _filter_score(orig_prediction: dict, score_thresh: float = 0.5):
    """Filter out predictions with confidence scores < score_thresh.

    :param orig_prediction: Original model prediction
    :type orig_prediction: dict
    :param score_thresh: Score threshold to filter by
    :type score_thresh: float
    :return: Model predictions filtered out by score_thresh
    :rtype: dict
    """
    keep = orig_prediction[SCORES] > score_thresh

    filter_prediction = orig_prediction
    filter_prediction[BOXES] = filter_prediction[BOXES][keep]
    filter_prediction[SCORES] = filter_prediction[SCORES][keep]
    filter_prediction[LABELS] = filter_prediction[LABELS][keep]
    return filter_prediction


def _apply_nms(orig_prediction: dict, iou_thresh: float = 0.5):
    """Perform nms on the predictions based on their IoU.

    :param orig_prediction: Original model prediction
    :type orig_prediction: dict
    :param iou_thresh: iou_threshold for nms
    :type iou_thresh: float
    :return: Model prediction after nms is applied
    :rtype: dict
    """
    keep = torchvision.ops.nms(orig_prediction[BOXES],
                               orig_prediction[SCORES],
                               iou_thresh)

    nms_prediction = orig_prediction
    nms_prediction[BOXES] = nms_prediction[BOXES][keep]
    nms_prediction[SCORES] = nms_prediction[SCORES][keep]
    nms_prediction[LABELS] = nms_prediction[LABELS][keep]
    return nms_prediction


def _wrap_image_model(model, examples, model_task, is_function, number_of_classes=None):
    """If needed, wraps the model or function in a common API.

    Wraps the model based on model task and prediction function contract.

    :param model: The model or function to evaluate on the examples.
    :type model: function or model to wrap
    :param examples: The model evaluation examples.
    :type examples: ml_wrappers.DatasetWrapper or numpy.ndarray
        or pandas.DataFrame or panads.Series or scipy.sparse.csr_matrix
        or shap.DenseData or torch.Tensor
    :param model_task: Parameter to specify whether the model is an
        'image_classification' or another type of image model.
    :type model_task: str
    :param number_of_classes: optional parameter specifying the number of classes in
        the dataset
    :type number_of_classes: int
    :return: The function chosen from given model and chosen domain, or
        model wrapping the function and chosen domain.
    :rtype: (function, str) or (model, str)
    """
    _wrapped_model = model
    if model_task == ModelTask.IMAGE_CLASSIFICATION:
        try:
            if isinstance(model, nn.Module):
                model = WrappedPytorchModel(model, image_to_tensor=True)
                if not isinstance(examples, DatasetWrapper):
                    examples = DatasetWrapper(examples)
                eval_function, eval_ml_domain = _eval_model(model, examples, model_task)
                return (
                    WrappedClassificationModel(model, eval_function, examples),
                    eval_ml_domain,
                )
        except (NameError, AttributeError):
            module_logger.debug(
                'Could not import torch, required if using a pytorch model'
            )

        if _is_fastai_model(model):
            _wrapped_model = WrappedFastAIImageClassificationModel(model)
        elif hasattr(model, '_model_impl'):
            if str(type(model._model_impl.python_model)).endswith(
                "azureml.automl.dnn.vision.common.mlflow.mlflow_model_wrapper.MLFlowImagesModelWrapper'>"
            ):
                _wrapped_model = WrappedMlflowAutomlImagesClassificationModel(model)
        else:
            _wrapped_model = WrappedTransformerImageClassificationModel(model)
    elif model_task == ModelTask.MULTILABEL_IMAGE_CLASSIFICATION:
        if _is_fastai_model(model):
            _wrapped_model = WrappedFastAIImageClassificationModel(
                model, multilabel=True
            )
    elif model_task == ModelTask.OBJECT_DETECTION:
        _wrapped_model = WrappedObjectDetectionModel(model, number_of_classes)
    return _wrapped_model, model_task


class WrappedTransformerImageClassificationModel(object):
    """A class for wrapping a Transformers model in the scikit-learn style."""

    def __init__(self, model):
        """Initialize the WrappedTransformerImageClassificationModel."""
        self._model = model

    def predict(self, dataset):
        """Predict the output using the wrapped Transformers model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        return np.argmax(self._model(dataset), axis=1)

    def predict_proba(self, dataset):
        """Predict the output probability using the Transformers model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        return self._model(dataset)


class WrappedFastAIImageClassificationModel(object):
    """A class for wrapping a FastAI model in the scikit-learn style."""

    def __init__(self, model, multilabel=False):
        """Initialize the WrappedFastAIImageClassificationModel.

        :param model: The model to wrap.
        :type model: fastai.learner.Learner
        :param multilabel: Whether the model is a multilabel model.
        :type multilabel: bool
        """
        self._model = model
        self._multilabel = multilabel

    def _fastai_predict(self, dataset, index):
        """Predict the output using the wrapped FastAI model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :param index: The index into the predicted data.
            Index 1 is for the predicted class and index
            2 is for the predicted probability.
        :type index: int
        :return: The predicted data.
        :rtype: numpy.ndarray
        """
        # Note predict for single image requires 3d instead of 4d array
        if len(dataset.shape) == 4:
            predictions = []
            for row in dataset:
                predictions.append(self._fastai_predict(row, index))
            predictions = np.array(predictions)
            if index == 1 and not self._multilabel:
                predictions = predictions.flatten()
            return predictions
        else:
            predictions = np.array(self._model.predict(dataset)[index])
            if len(predictions.shape) == 0:
                predictions = predictions.reshape(1)
            if index == 1:
                is_boolean = predictions.dtype == bool
                if is_boolean:
                    predictions = predictions.astype(int)
            return predictions

    def predict(self, dataset):
        """Predict the output value using the wrapped FastAI model.

        :param dataset: The dataset to predict on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        return self._fastai_predict(dataset, 1)

    def predict_proba(self, dataset):
        """Predict the output probability using the FastAI model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        return self._fastai_predict(dataset, 2)


class WrappedMlflowAutomlImagesClassificationModel:
    """A class for wrapping an AutoML for images MLflow model in the scikit-learn style."""

    def __init__(self, model: PyFuncModel) -> None:
        """Initialize the WrappedMlflowAutomlImagesClassificationModel.

        :param model: mlflow model
        :type model: mlflow.pyfunc.PyFuncModel
        """
        self._model = model

    def _mlflow_predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Perform the inference using the wrapped MLflow model.

        :param dataset: The dataset to predict on.
        :type dataset: pandas.DataFrame
        :return: The predicted data.
        :rtype: pandas.DataFrame
        """
        predictions = self._model.predict(dataset)
        return predictions

    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        """Predict the output value using the wrapped MLflow model.

        :param dataset: The dataset to predict on.
        :type dataset: pandas.DataFrame
        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        predictions = self._mlflow_predict(dataset)
        return predictions.loc[:, 'probs'].map(lambda x: np.argmax(x)).values

    def predict_proba(self, dataset: pd.DataFrame) -> np.ndarray:
        """Predict the output probability using the MLflow model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: pandas.DataFrame
        :return: The predicted probabilities.
        :rtype: numpy.ndarray
        """
        predictions = self._mlflow_predict(dataset)
        return np.stack(predictions.probs.values)


class WrappedObjectDetectionModel:
    """A class for wrapping a object detection model in the scikit-learn style."""

    def __init__(self, model: Any, number_of_classes: int) -> None:
        """Initialize the WrappedObjectDetectionModel with the model and evaluation function.

        :param model: mlflow model
        :type model: Any
        """
        self._device = torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
        model.eval()
        self._model = model
        self._number_of_classes = number_of_classes

    def predict(self, x, iou_thresh: float = 0.5, score_thresh: float = 0.5):
        """Create a list of detection records from the image predictions.

        :param x: Tensor of the image
        :type x: torch.Tensor
        :return: Baseline detections to get saliency maps for
        :rtype: List of Detection Records

        Example Label (y) representation for a cohort of 2 images:

        [

            [
            [object_1, x1, y1, b1, h1, (optional) confidence_score],
            [object_2, x2, y2, b2, h2, (optional) confidence_score],
            [object_1, x3, y3, b3, h3, (optional) confidence_score]
        ],

        [
            [object_1, x4, y4, b4, h4, (optional) confidence_score],
            [object_2, x5, y5, b5, h5, (optional) confidence_score]
        ]

        ]

        """
        detections = []
        for image in x:
            if type(image) == torch.Tensor:
                raw_detections = self._model(
                    image.to(self._device).unsqueeze(0))
            else:
                raw_detections = self._model(
                    T.ToTensor()(image).to(self._device).unsqueeze(0))

            for raw_detection in raw_detections:
                raw_detection = _apply_nms(raw_detection)
                raw_detection = _filter_score(raw_detection)
                image_predictions = torch.cat((raw_detection["labels"]
                                               .unsqueeze(1),
                                               raw_detection["boxes"],
                                               raw_detection["scores"]
                                               .unsqueeze(1)), dim=1)

                detections.append(image_predictions.detach().cpu().numpy()
                                  .tolist())
        return detections

    def predict_proba(self, dataset, iou_threshold=0.1):
        """Predict the output probability using the wrapped model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: ml_wrappers.DatasetWrapper
        param iou_threshold: amount of acceptable error.
            objects with error scores higher than the threshold will be removed
        type iou_threshold: float
        """
        predictions = self.predict(dataset, iou_threshold)
        prob_scores = [[pred.class_scores for pred in image_prediction] for image_prediction in predictions]
        return prob_scores


class PytorchDRiseWrapper(
        od_common.GeneralObjectDetectionModelWrapper):
    """Wraps a PytorchFasterRCNN model with a predict API function.

    To be compatible with the D-RISE explainability method,
    all models must be wrapped to have the same output and input class and a
    predict function for object detection. This wrapper is customized for the
    FasterRCNN model from Pytorch, and can also be used with the RetinaNet or
    any other models with the same output class.
    """

    def __init__(self, model, number_of_classes: int):
        """Initialize the PytorchDRiseWrapper.

        :param model: Object detection model
        :type model: PytorchFasterRCNN model
        :param number_of_classes: Number of classes the model is predicting
        :type number_of_classes: int
        """
        model.eval()
        self._model = model
        self._number_of_classes = number_of_classes

    def predict(self, x: torch.Tensor):
        """Create a list of detection records from the image predictions.

        :param x: Tensor of the image
        :type x: torch.Tensor
        :return: Baseline detections to get saliency maps for
        :rtype: List of Detection Records
        """
        raw_detections = self._model(x)

        detections = []
        for raw_detection in raw_detections:
            raw_detection = _apply_nms(raw_detection, 0.005)

            # Note that FasterRCNN doesn't return a score for each class, only
            # the predicted class. DRISE requires a score for each class.
            # We approximate the score for each class
            # by dividing (class score) evenly among the other classes.

            raw_detection = _filter_score(raw_detection, 0.5)
            expanded_class_scores = od_common.expand_class_scores(
                raw_detection[SCORES],
                raw_detection[LABELS],
                self._number_of_classes)

            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detection[BOXES],
                    class_scores=expanded_class_scores,
                    objectness_scores=torch.tensor(
                        [1.0]*raw_detection[BOXES].shape[0]),
                )
            )
        return detections
