# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the EndpointWrapperModel class."""

import sys
from unittest.mock import patch

import openai
import pandas as pd
import pytest
from ml_wrappers.model import OpenaiWrapperModel
from openai.types.chat.chat_completion import (ChatCompletion,
                                               ChatCompletionMessage, Choice,
                                               CompletionUsage)

CHOICES = 'choices'
MESSAGE = 'message'
CONTENT = 'content'


@pytest.mark.usefixtures('_clean_dir')
class TestOpenaiWrapperModel(object):
    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call(self):
        # test creating the OpenaiWrapperModel and
        # calling the predict function
        api_type = 'azure'
        api_base = 'https://mock.openai.azure.com/'
        api_version = '2023-03-15-preview'
        api_key = 'mock'
        context = ''
        questions = 'How to convert 10^9/l to liter?'
        is_openai_1_0 = False
        if hasattr(openai, 'ChatCompletion'):
            mock_function = 'openai.ChatCompletion.create'
        else:
            mock_function = 'openai.resources.chat.completions.Completions.create'
            is_openai_1_0 = True
        answer = (' It seems there is some confusion with the units being'
                  ' used in your question. The symbol `10^9/l` is often'
                  ' used to represent a concentration of 10^9 molecules'
                  ' or particles per liter of a solution. However, it is'
                  ' not a unit of volume and cannot be directly converted'
                  ' to liters. If you are trying to convert a concentration'
                  ' from one unit to another, such as from micrograms per'
                  ' liter (µg/L) to milligrams per liter (mg/L), you can'
                  ' use the appropriate conversion factor. For example,'
                  ' 1 µg/L is equal to 0.001 mg/L. If you need help'
                  ' with a specific conversion, please provide more'
                  ' details and I will do my best to assist you.')
        test_data = pd.DataFrame(data=[[context, questions, answer]],
                                 columns=['context', 'questions', 'answer'])
        expected_content = ('To convert from 10^9 per liter to '
                            'liters, you need to find the '
                            'reciprocal of the given value.'
                            '\n\nReciprocal of 10^9 per '
                            'liter = 1 / (10^9 per liter)'
                            '\n\nSo, the value in liters '
                            'is 1 / 10^9 liters, or '
                            '10^(-9) liters.')
        expected_model = 'gpt-4-32k'
        expected_id = 'chatcmpl-XYZ'
        expected_object = 'chat.completion'
        expected_created_stamp = 123
        expected_role = 'assistant'
        expected_finish_reason = 'stop'
        expected_completion_tokens = 69
        expected_prompt_tokens = 18
        expected_total_tokens = 87
        if is_openai_1_0:
            mock_result = ChatCompletion(
                id=expected_id,
                choices=[Choice(
                    finish_reason=expected_finish_reason,
                    index=0,
                    message=ChatCompletionMessage(content=expected_content,
                                                  role=expected_role,
                                                  function_call=None))],
                created=expected_created_stamp,
                model=expected_model,
                object=expected_object,
                usage=CompletionUsage(
                    completion_tokens=expected_completion_tokens,
                    prompt_tokens=expected_prompt_tokens,
                    total_tokens=expected_total_tokens))
        else:
            mock_result = {
                'id': expected_id,
                'object': expected_object,
                'created': expected_created_stamp,
                'model': expected_model,
                CHOICES: [
                    {
                        'index': 0,
                        'finish_reason': expected_finish_reason,
                        MESSAGE: {
                            'role': expected_role,
                            CONTENT: expected_content
                        }
                    }
                ],
                'usage': {
                    'completion_tokens': expected_completion_tokens,
                    'prompt_tokens': expected_prompt_tokens,
                    'total_tokens': expected_total_tokens
                }
            }
        openai_model = OpenaiWrapperModel(
            api_type, api_base, api_version, api_key)
        # mock the openai create function
        with patch(mock_function) as mock_create:
            # wrap return value in mock class with read method
            mock_create.return_value = mock_result
            context = {}
            result = openai_model.predict(context, test_data)
            if is_openai_1_0:
                expected_result = mock_result.choices[0].message.content
            else:
                expected_result = mock_result[CHOICES][0][MESSAGE][CONTENT]
            assert len(result) == 1
            assert result[0] == expected_result
