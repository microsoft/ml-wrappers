# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on vision-based models"""

import sys

import pytest
from common_vision_utils import (create_image_classification_pipeline,
                                 load_fridge_dataset, load_imagenet_dataset,
                                 load_images, retrieve_or_train_fridge_model)
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
