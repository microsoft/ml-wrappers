# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the EndpointWrapperModel class."""

import sys
from unittest.mock import patch

import pandas as pd
import pytest
from ml_wrappers.model import OpenaiWrapperModel


@pytest.mark.usefixtures('_clean_dir')
class TestOpenaiWrapperModel(object):
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call(self):
        # test creating the OpenaiWrapperModel and
        # calling the predict function
        api_type = "azure"
        api_base = "https://mock.openai.azure.com/"
        api_version = "2023-03-15-preview"
        api_key = "mock"
        context = ''
        questions = "How to convert 10^9/l to liter?"
        answer = (' It seems there is some confusion with the units being' +
                  ' used in your question. The symbol `10^9/l` is often' +
                  ' used to represent a concentration of 10^9 molecules' +
                  ' or particles per liter of a solution. However, it is' +
                  ' not a unit of volume and cannot be directly converted' +
                  ' to liters. If you are trying to convert a concentration' +
                  ' from one unit to another, such as from micrograms per' +
                  ' liter (µg/L) to milligrams per liter (mg/L), you can' +
                  ' use the appropriate conversion factor. For example,' +
                  ' 1 µg/L is equal to 0.001 mg/L. If you need help' +
                  ' with a specific conversion, please provide more' +
                  ' details and I will do my best to assist you.')
        test_data = pd.DataFrame(data=[[context, questions, answer]],
                                 columns=['context', 'questions', 'answer'])
        mock_result = {
            "id": "chatcmpl-XYZ",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4-32k",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": ('To convert from 10^9 per liter to ' +
                                    'liters, you need to find the ' +
                                    'reciprocal of the given value.' +
                                    '\n\nReciprocal of 10^9 per ' +
                                    'liter = 1 / (10^9 per liter)' +
                                    '\n\nSo, the value in liters ' +
                                    'is 1 / 10^9 liters, or ' +
                                    '10^(-9) liters.')
                    }
                }
            ],
            "usage": {
                "completion_tokens": 69,
                "prompt_tokens": 18,
                "total_tokens": 87
            }
        }
        openai_model = OpenaiWrapperModel(
            api_type, api_base, api_version, api_key)
        # mock the openai.ChatCompletion.create function
        with patch('openai.ChatCompletion.create') as mock_create:
            # wrap return value in mock class with read method
            mock_create.return_value = mock_result
            context = {}
            result = openai_model.predict(context, test_data)
            expected_result = mock_result["choices"][0]["message"]["content"]
            assert len(result) == 1
            assert result[0] == expected_result
