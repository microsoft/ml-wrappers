# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on vision-based models"""

import pytest
from common_vision_utils import (create_image_classification_pipeline,
                                 load_imagenet_dataset)
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
