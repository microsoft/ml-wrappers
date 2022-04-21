# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on text-based models"""

import pytest
from common_text_utils import (EMOTION, create_question_answering_pipeline,
                               create_text_classification_pipeline,
                               load_emotion_dataset, load_squad_dataset)
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask
from wrapper_validator import (validate_wrapped_classification_model,
                               validate_wrapped_question_answering_model)


@pytest.mark.usefixtures('_clean_dir')
class TestTextModelWrapper(object):
    def test_wrap_transformers_model(self):
        emotion_data = load_emotion_dataset()
        docs = emotion_data[:10].drop(columns=EMOTION).values.tolist()
        pred = create_text_classification_pipeline()
        wrapped_model = wrap_model(pred, docs, ModelTask.TEXT_CLASSIFICATION)
        validate_wrapped_classification_model(wrapped_model, docs)

    def test_wrap_question_answering_model(self):
        squad_data = load_squad_dataset()
        docs = squad_data[:10].drop(columns=['answers'])
        pred = create_question_answering_pipeline()
        wrapped_model = wrap_model(pred, docs, ModelTask.QUESTION_ANSWERING)
        validate_wrapped_question_answering_model(wrapped_model, docs)
