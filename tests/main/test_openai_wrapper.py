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

try:
    from openai.types.chat.chat_completion import (ChatCompletion,
                                                   ChatCompletionMessage,
                                                   Choice, CompletionUsage)
except ImportError:
    # Ignore the error, only used by new openai version
    pass

CHOICES = 'choices'
MESSAGE = 'message'
CONTENT = 'content'


@pytest.mark.usefixtures('_clean_dir')
class TestOpenaiWrapperModel(object):
    QUESTION = 'How to convert 10^9/l to liter?'
    ANSWER = (' It seems there is some confusion with the units being'
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
    EXPECTED_ANSWER = ('To convert from 10^9 per liter to '
                       'liters, you need to find the '
                       'reciprocal of the given value.'
                       '\n\nReciprocal of 10^9 per '
                       'liter = 1 / (10^9 per liter)'
                       '\n\nSo, the value in liters '
                       'is 1 / 10^9 liters, or '
                       '10^(-9) liters.')
    SYS_PROMPT = 'You are a helpful assistant.'
    HISTORY = [
        {'role': 'user', 'content': 'How are you?'},
        {'role': 'assistant', 'content': 'I am good. How can I help you?'}
    ]

    def create_mock_result(self, expected_content, is_openai_1_0: bool):
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
        return mock_result

    def get_model_params(self):
        ret = {
            'context': '',
            'expected_model': 'gpt-4-32k'
        }
        api_type = 'azure'
        api_base = 'https://mock.openai.azure.com/'
        api_version = '2023-03-15-preview'
        api_key = 'mock'
        ret['openai_model'] = OpenaiWrapperModel(
            api_type, api_base, api_version, api_key)
        is_openai_1_0 = False
        if not hasattr(openai, 'OpenAI'):
            mock_function = 'openai.ChatCompletion.create'
        else:
            mock_function = 'openai.resources.chat.completions.Completions.create'
            is_openai_1_0 = True
        ret['is_openai_1_0'] = is_openai_1_0
        ret['mock_function'] = mock_function
        return ret

    def assert_result(self, params, mock_result, test_data):
        # mock the openai create function
        with patch(params['mock_function']) as mock_create:
            # wrap return value in mock class with read method
            mock_create.return_value = mock_result
            context = {}
            result = params['openai_model'].predict(context, test_data)
            if params['is_openai_1_0']:
                expected_result = mock_result.choices[0].message.content
            else:
                expected_result = mock_result[CHOICES][0][MESSAGE][CONTENT]
            assert len(result) == 1
            assert result[0] == expected_result

    def assert_called_with(self, params, mock_result, test_data, **kwargs):
        # mock the openai create function
        with patch(params['mock_function']) as mock_create:
            # wrap return value in mock class with read method
            mock_create.return_value = mock_result
            context = {}
            params['openai_model'].predict(context, test_data)
            messages = []
            if 'sys_prompt' in kwargs:
                messages.append({'role': 'system', 'content': kwargs['sys_prompt']})
            if 'history' in kwargs:
                messages.extend(kwargs['history'])
            messages.append({'role': 'user', 'content': kwargs['questions']})
            if params['is_openai_1_0']:
                mock_create.assert_called_with(
                    model=params['expected_model'],
                    messages=messages,
                    temperature=params['openai_model'].temperature,
                    max_tokens=params['openai_model'].max_tokens,
                    top_p=params['openai_model'].top_p,
                    frequency_penalty=params['openai_model'].frequency_penalty,
                    presence_penalty=params['openai_model'].presence_penalty,
                    stop=params['openai_model'].stop
                )
            else:
                mock_create.assert_called_with(
                    engine=params['expected_model'],
                    messages=messages,
                    temperature=params['openai_model'].temperature,
                    max_tokens=params['openai_model'].max_tokens,
                    top_p=params['openai_model'].top_p,
                    frequency_penalty=params['openai_model'].frequency_penalty,
                    presence_penalty=params['openai_model'].presence_penalty,
                    stop=params['openai_model'].stop
                )

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call(self):
        params = self.get_model_params()
        questions = self.QUESTION
        answer = self.ANSWER
        test_data = pd.DataFrame(data=[[params['context'], questions, answer]],
                                 columns=['context', 'questions', 'answer'])
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        self.assert_result(params, mock_result, test_data)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_with_sys_prompt(self):
        params = self.get_model_params()
        questions = self.QUESTION
        sys_prompt = self.SYS_PROMPT
        answer = self.ANSWER
        test_data = pd.DataFrame(data=[[params['context'], questions, answer, sys_prompt]],
                                 columns=['context', 'questions', 'answer', 'sys_prompt'])
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])
        self.assert_result(params, mock_result, test_data)
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_with_history(self):
        params = self.get_model_params()
        questions = self.QUESTION
        history = self.HISTORY
        answer = self.ANSWER
        test_data = pd.DataFrame(data=[[params['context'], questions, answer, history]],
                                 columns=['context', 'questions', 'answer', 'history'])
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])
        self.assert_result(params, mock_result, test_data)
        self.assert_called_with(params, mock_result, test_data,
                                history=history, questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_with_sys_prompt_and_history(self):
        params = self.get_model_params()
        questions = self.QUESTION
        sys_prompt = self.SYS_PROMPT
        history = self.HISTORY
        answer = self.ANSWER
        test_data = pd.DataFrame(data=[[params['context'], questions, answer, sys_prompt, history]],
                                 columns=['context', 'questions', 'answer', 'sys_prompt', 'history'])
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])
        self.assert_result(params, mock_result, test_data)
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, history=history,
                                questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_df(self):
        params = self.get_model_params()
        questions = self.QUESTION
        sys_prompt = self.SYS_PROMPT
        history = self.HISTORY
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        # Only prompt
        test_data = pd.DataFrame(data=[[params['context'], questions]],
                                 columns=['context', 'questions'])
        self.assert_called_with(params, mock_result, test_data,
                                questions=questions)

        # Prompt and sys_prompt
        test_data = pd.DataFrame(data=[[params['context'], questions, sys_prompt]],
                                 columns=['context', 'questions', 'sys_prompt'])
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, questions=questions)

        # Prompt and history
        test_data = pd.DataFrame(data=[[params['context'], questions, history]],
                                 columns=['context', 'questions', 'history'])
        self.assert_called_with(params, mock_result, test_data,
                                history=history, questions=questions)

        # Prompt, history, and sys_prompt
        test_data = pd.DataFrame(data=[[params['context'], questions, history, sys_prompt]],
                                 columns=['context', 'questions', 'history', 'sys_prompt'])
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, history=history,
                                questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_dict(self):
        params = self.get_model_params()
        questions = self.QUESTION
        sys_prompt = self.SYS_PROMPT
        history = self.HISTORY
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        # Only prompt
        test_data = {
            'context': [params['context']],
            'questions': [questions]
        }
        self.assert_called_with(params, mock_result, test_data,
                                questions=questions)

        # Prompt and sys_prompt
        test_data = {
            'context': [params['context']],
            'questions': [questions],
            'sys_prompt': [sys_prompt]
        }
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, questions=questions)

        # Prompt and history
        test_data = {
            'context': [params['context']],
            'questions': [questions],
            'history': [history]
        }
        self.assert_called_with(params, mock_result, test_data,
                                history=history, questions=questions)

        # Prompt, history, and sys_prompt
        test_data = {
            'context': [params['context']],
            'questions': [questions],
            'history': [history],
            'sys_prompt': [sys_prompt]
        }
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, history=history,
                                questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_dict_of_str(self):
        params = self.get_model_params()
        questions = self.QUESTION
        sys_prompt = self.SYS_PROMPT
        history = self.HISTORY
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        # Only prompt
        test_data = {
            'context': params['context'],
            'questions': questions
        }
        self.assert_called_with(params, mock_result, test_data,
                                questions=questions)

        # Prompt and sys_prompt
        test_data = {
            'context': params['context'],
            'questions': questions,
            'sys_prompt': sys_prompt
        }
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, questions=questions)

        # Prompt and history
        test_data = {
            'context': params['context'],
            'questions': questions,
            'history': history
        }
        self.assert_called_with(params, mock_result, test_data,
                                history=history, questions=questions)

        # Prompt, history, and sys_prompt
        test_data = {
            'context': params['context'],
            'questions': questions,
            'history': history,
            'sys_prompt': sys_prompt
        }
        self.assert_called_with(params, mock_result, test_data,
                                sys_prompt=sys_prompt, history=history,
                                questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_str(self):
        params = self.get_model_params()
        questions = self.QUESTION
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        # Only prompt
        test_data = questions
        self.assert_called_with(params, mock_result, test_data,
                                questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_list_of_str(self):
        params = self.get_model_params()
        questions = self.QUESTION
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        # Only prompt
        test_data = [questions]
        self.assert_called_with(params, mock_result, test_data,
                                questions=questions)

    @pytest.mark.skipif(sys.version_info.minor <= 6,
                        reason='Openai not supported for older versions')
    def test_predict_call_series(self):
        params = self.get_model_params()
        questions = self.QUESTION
        expected_content = self.EXPECTED_ANSWER
        mock_result = self.create_mock_result(expected_content, params['is_openai_1_0'])

        # Only prompt
        test_data = pd.Series(questions)
        self.assert_called_with(params, mock_result, test_data,
                                questions=questions)
