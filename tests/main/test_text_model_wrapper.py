# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on text-based models"""

import pytest
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask

from common_text_utils import (EMOTION, create_multilabel_text_pipeline,
                               create_question_answering_pipeline,
                               create_text_classification_pipeline,
                               load_covid19_emergency_event_dataset,
                               load_emotion_dataset, load_squad_dataset)
from wrapper_validator import (validate_wrapped_classification_model,
                               validate_wrapped_multilabel_model,
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

    def test_wrap_multilabel_model(self):
        covid19_data = load_covid19_emergency_event_dataset()
        docs = covid19_data[:10]['text'].values.tolist()
        pred = create_multilabel_text_pipeline()
        wrapped_model = wrap_model(
            pred, docs, ModelTask.MULTILABEL_TEXT_CLASSIFICATION)
        num_labels = pred.model.num_labels
        validate_wrapped_multilabel_model(wrapped_model, docs, num_labels)
