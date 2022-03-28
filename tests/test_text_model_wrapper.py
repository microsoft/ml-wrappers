# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on text-based models"""

import pytest
from common_text_utils import (EMOTION, create_text_classification_pipeline,
                               load_emotion_dataset)
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask
from wrapper_validator import validate_wrapped_classification_model


@pytest.mark.usefixtures('clean_dir')
class TestTextModelWrapper(object):
    def test_wrap_transformers_model(self, iris):
        emotion_data = load_emotion_dataset()
        docs = emotion_data[:10].drop(columns=EMOTION).values.tolist()
        pred = create_text_classification_pipeline()
        wrapped_model = wrap_model(pred, docs, ModelTask.TEXT_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, docs)
