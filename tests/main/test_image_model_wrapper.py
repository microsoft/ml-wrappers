# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on vision-based models"""

import sys

import numpy as np
import pandas as pd
import pytest
import torchvision
from common_vision_utils import (IMAGE, create_image_classification_pipeline,
                                 create_pytorch_image_model,
                                 create_scikit_classification_pipeline,
                                 load_fridge_dataset, load_imagenet_dataset,
                                 load_images, load_multilabel_fridge_dataset,
                                 load_object_fridge_dataset,
                                 preprocess_imagenet_dataset,
                                 retrieve_or_train_fridge_model)
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask
from ml_wrappers.model.image_model_wrapper import (PytorchDRiseWrapper,
                                                   WrappedObjectDetectionModel,
                                                   _apply_nms, _get_device)
from wrapper_validator import (validate_wrapped_classification_model,
                               validate_wrapped_multilabel_model,
                               validate_wrapped_object_detection_custom_model,
                               validate_wrapped_object_detection_model)

try:
    import torch
    from torchvision import transforms as T
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except ImportError:
    print('Could not import torch, required if using a PyTorch model')


NUM_FRIDGE_CLASSES = 5
NUM_TEST_IMAGES = 3


class CustomObjectDetectionWrapper(WrappedObjectDetectionModel):
    def __init__(self):
        model = _set_up_OD_model()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        super(CustomObjectDetectionWrapper, self).__init__(
            model, NUM_FRIDGE_CLASSES, device)

    def predict(self, dataset, iou_threshold=0.5, score_threshold=0.5):
        return super(CustomObjectDetectionWrapper, self).predict(
            dataset, iou_threshold, score_threshold)

    def predict_proba(self, dataset):
        return super(CustomObjectDetectionWrapper, self).predict_proba(dataset)


def _set_up_OD_data(num_images=1):
    """Returns generic dataset for OD testing (FastRCNN)"""
    data = load_object_fridge_dataset()[:num_images]
    data = load_images(data)
    return data


def _set_up_OD_model():
    """Returns generic model for OD testing (FastRCNN)"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_FRIDGE_CLASSES)
    return model


def _set_up_OD_model_data(num_images=1):
    """Returns generic model and dataset for OD testing (FastRCNN)"""
    data = _set_up_OD_data(num_images)
    model = _set_up_OD_model()
    return model, data


@pytest.mark.usefixtures('_clean_dir')
class TestImageModelWrapper(object):
    def test_wrap_resnet_classification_model(self):
        data = load_imagenet_dataset()
        pred = create_image_classification_pipeline()
        wrapped_model = wrap_model(pred, data, ModelTask.IMAGE_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, data)

    def test_wrap_scikit_classification_model(self):
        data = load_imagenet_dataset()
        pred = create_scikit_classification_pipeline()
        wrapped_model = wrap_model(pred, data, ModelTask.IMAGE_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, data)

    # Skip for older versions of python due to many breaking changes in fastai
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Fastai not supported for older versions')
    # Skip is using macos due to fastai failing on latest macos
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='Fastai not supported for latest macos')
    def test_wrap_fastai_image_classification_model(self):
        data = load_fridge_dataset()
        try:
            model = retrieve_or_train_fridge_model(data)
        except Exception as e:
            print("Failed to retrieve or load Fastai model, force training")
            print("Inner exception message on retrieving model: {}".format(e))
            model = retrieve_or_train_fridge_model(data, force_train=True)
        # load the paths as numpy arrays
        data = load_images(data)
        wrapped_model = wrap_model(model, data, ModelTask.IMAGE_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, data)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_pytorch_image_classification_model(self):
        data = load_imagenet_dataset()[:3]
        data = preprocess_imagenet_dataset(data)
        model = create_pytorch_image_model()
        wrapped_model = wrap_model(model, data, ModelTask.IMAGE_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, data)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_pytorch_image_classification_model_pandas(self):
        data = load_imagenet_dataset()[:3]
        data = preprocess_imagenet_dataset(data)
        # convert to pandas dataframe of images
        data = np.array(data)
        # change back to channel at last dimension for
        # standard image representation
        data = np.moveaxis(data, 1, -1)
        imgs = np.empty((len(data)), dtype=object)
        for i, row in enumerate(data):
            imgs[i] = row
        data = pd.DataFrame(imgs, columns=[IMAGE])
        model = create_pytorch_image_model()
        wrapped_model = wrap_model(model, data, ModelTask.IMAGE_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, data)

    # Skip for older versions of python due to many breaking changes in fastai
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Fastai not supported for older versions')
    # Skip is using macos due to fastai failing on latest macos
    @pytest.mark.skipif(sys.platform == 'darwin',
                        reason='Fastai not supported for latest macos')
    def test_wrap_fastai_multilabel_image_classification_model(self):
        data = load_multilabel_fridge_dataset()
        try:
            model = retrieve_or_train_fridge_model(data, multilabel=True)
        except Exception as e:
            print("Failed to retrieve or load Fastai model, force training")
            print("Inner exception message on retrieving model: {}".format(e))
            model = retrieve_or_train_fridge_model(
                data, force_train=True, multilabel=True)
        # load the paths as numpy arrays
        data = load_images(data)
        wrapped_model = wrap_model(
            model, data, ModelTask.MULTILABEL_IMAGE_CLASSIFICATION)
        num_labels = 4
        validate_wrapped_multilabel_model(wrapped_model, data, num_labels)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_pytorch_object_detection_model_pandas(self):
        model, data = _set_up_OD_model_data(NUM_TEST_IMAGES)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wrapped_model = wrap_model(model.to(device),
                                   data,
                                   ModelTask.OBJECT_DETECTION)
        validate_wrapped_object_detection_model(wrapped_model, data)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_pytorch_object_detection_custom_model_pandas(self):
        model, data = _set_up_OD_model_data()
        wrapped_model = PytorchDRiseWrapper(model, 1)
        validate_wrapped_object_detection_custom_model(wrapped_model,
                                                       T.ToTensor()(data[0])
                                                       .repeat(2, 1, 1, 1),
                                                       has_predict_proba=False)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_custom_object_detection_wrapper(self):
        data = _set_up_OD_data(NUM_TEST_IMAGES)
        model = CustomObjectDetectionWrapper()
        wrapped_model = wrap_model(model,
                                   data,
                                   ModelTask.OBJECT_DETECTION)
        assert isinstance(wrapped_model, CustomObjectDetectionWrapper)
        validate_wrapped_object_detection_model(wrapped_model, data)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_PytorchDRiseWrapper_wrapper_device(self):
        model, data = _set_up_OD_model_data()

        wrapped_model = PytorchDRiseWrapper(model, 1, 'cpu')
        validate_wrapped_object_detection_custom_model(wrapped_model,
                                                       T.ToTensor()(data[0])
                                                       .repeat(2, 1, 1, 1),
                                                       has_predict_proba=False)
        if torch.cuda.is_available():
            wrapped_model = PytorchDRiseWrapper(model, 1, 'cuda')
            validate_wrapped_object_detection_custom_model(
                wrapped_model,
                T.ToTensor()(data[0])
                .repeat(2, 1, 1, 1),
                has_predict_proba=False)
        else:
            with pytest.raises(AssertionError,
                               match="Torch not compiled with CUDA enabled"):
                wrapped_model = PytorchDRiseWrapper(model, 1, 'cuda')

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_WrappedObjectDetectionModel_wrapper_device(self):
        model, data = _set_up_OD_model_data()
        wrapped_model = WrappedObjectDetectionModel(model, 1, 'cpu')
        validate_wrapped_object_detection_custom_model(wrapped_model,
                                                       T.ToTensor()(data[0])
                                                       .repeat(2, 1, 1, 1))
        if torch.cuda.is_available():
            wrapped_model = WrappedObjectDetectionModel(model, 1, 'cuda')
            validate_wrapped_object_detection_custom_model(
                wrapped_model,
                T.ToTensor()(data[0])
                .repeat(2, 1, 1, 1))
        else:
            with pytest.raises(AssertionError,
                               match="Torch not compiled with CUDA enabled"):
                wrapped_model = WrappedObjectDetectionModel(model, 1, 'cuda')

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    @pytest.mark.parametrize("boxes, scores, labels, iou_threshold, expected_boxes, expected_scores, expected_labels", [
        (torch.empty((0, 4)), torch.tensor([]), torch.tensor([]), 0.5, torch.empty((0, 4)), torch.tensor([]), torch.tensor([])),
        (torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]), torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 2, 3]), 0.5, torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]), torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 2, 3])),
        (torch.tensor([[0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2]]), torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 2, 3]), 0.5, torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]), torch.tensor([0.9, 0.7]), torch.tensor([1, 3])),
    ])
    def test_apply_nms(self, boxes, scores, labels, iou_threshold, expected_boxes, expected_scores, expected_labels):
        # Create the input dictionary
        orig_prediction = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }

        # Call the function being tested
        nms_prediction = _apply_nms(orig_prediction, iou_threshold)

        # Check that the output is as expected
        assert torch.all(torch.eq(nms_prediction['boxes'], expected_boxes))
        assert torch.all(torch.eq(nms_prediction['scores'], expected_scores))
        assert torch.all(torch.eq(nms_prediction['labels'], expected_labels))

    def test_get_device(self):
        # test default invocation of _get_device as it would be during the
        # wrap_model invocation in RAIVisionInsights
        device = _get_device("auto")
        assert device == "cpu" or device == "cuda"
        device = _get_device("cuda:1")
        assert device == "cuda:1"
        device = _get_device("cpu")
        assert device == "cpu"
