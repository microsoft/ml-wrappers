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
                                                   WrappedObjectDetectionModel)
from wrapper_validator import (validate_wrapped_classification_model,
                               validate_wrapped_multilabel_model,
                               validate_wrapped_object_detection_custom_model,
                               validate_wrapped_object_detection_model)

try:
    import torch
    from torchvision import transforms as T
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except ImportError:
    print('Could not import torchvision, required if using a vision PyTorch model')


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
        data = load_object_fridge_dataset()[:3]
        data = load_images(data)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wrapped_model = wrap_model(model.to(device),
                                   data,
                                   ModelTask.OBJECT_DETECTION)
        validate_wrapped_object_detection_model(wrapped_model, data)

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_pytorch_object_detection_custom_model_pandas(self):
        model, data = self._set_up_OD_model()
        wrapped_model = PytorchDRiseWrapper(model, 1)
        validate_wrapped_object_detection_custom_model(wrapped_model,
                                                       T.ToTensor()(data[0])
                                                       .repeat(2, 1, 1, 1))

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_PytorchDRiseWrapper_wrapper_device(self):
        model, data = self._set_up_OD_model()

        wrapped_model = PytorchDRiseWrapper(model, 1, 'cpu')
        validate_wrapped_object_detection_custom_model(wrapped_model,
                                                       T.ToTensor()(data[0])
                                                       .repeat(2, 1, 1, 1))
        if (torch.cuda.is_available()):
            wrapped_model = PytorchDRiseWrapper(model, 1, 'cuda')
            validate_wrapped_object_detection_custom_model(
                wrapped_model,
                T.ToTensor()(data[0])
                .repeat(2, 1, 1, 1))
        else:
            with pytest.raises(ValueError("Selected device is invalid")):
                wrapped_model = PytorchDRiseWrapper(model, 1, 'cuda')
                validate_wrapped_object_detection_custom_model(
                    wrapped_model,
                    T.ToTensor()(data[0])
                    .repeat(2, 1, 1, 1))

    # Skip for older versions of pytorch due to missing classes
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Older versions of pytorch not supported')
    def test_WrappedObjectDetectionModel_wrapper_device(self):
        model, data = self._set_up_OD_model()
        wrapped_model = WrappedObjectDetectionModel(model, 1, 'cpu')
        validate_wrapped_object_detection_custom_model(wrapped_model,
                                                       T.ToTensor()(data[0])
                                                       .repeat(2, 1, 1, 1))
        if (torch.cuda.is_available()):
            wrapped_model = WrappedObjectDetectionModel(model, 1, 'cuda')
            validate_wrapped_object_detection_custom_model(
                wrapped_model,
                T.ToTensor()(data[0])
                .repeat(2, 1, 1, 1))
        else:
            with pytest.raises(ValueError("Selected device is invalid")):
                wrapped_model = WrappedObjectDetectionModel(model, 1, 'cuda')
                validate_wrapped_object_detection_custom_model(
                    wrapped_model,
                    T.ToTensor()(data[0])
                    .repeat(2, 1, 1, 1))

    def _set_up_OD_model(self):
        """Returns generic model and dataset for OD testing (FastRCNN)"""
        data = load_object_fridge_dataset()[:1]
        data = load_images(data)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
        return model, data
