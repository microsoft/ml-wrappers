# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on vision-based models"""

import sys

import numpy as np
import pandas as pd
import pytest
from common_vision_utils import (IMAGE, create_image_classification_pipeline,
                                 create_pytorch_image_model,
                                 load_fridge_dataset, load_imagenet_dataset,
                                 load_images, preprocess_imagenet_dataset,
                                 retrieve_or_train_fridge_model)
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask
from wrapper_validator import validate_wrapped_classification_model


@pytest.mark.usefixtures('_clean_dir')
class TestImageModelWrapper(object):
    def test_wrap_resnet_classification_model(self):
        data = load_imagenet_dataset()
        pred = create_image_classification_pipeline()
        wrapped_model = wrap_model(pred, data, ModelTask.IMAGE_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, data)

    # Skip for older versions of python due to many breaking changes in fastai
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Fastai not supported for older versions')
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