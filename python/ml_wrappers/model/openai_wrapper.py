# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a model wrapper for an openai model endpoint."""

import numpy as np

try:
    import openai
    openai_installed = True
except ImportError:
    openai_installed = False
try:
    from raiutils.common.retries import retry_function
    rai_utils_installed = True
except ImportError:
    rai_utils_installed = False


CONTENT = 'content'


def replace_backtick_chars(message):
    """Replace backtick characters in a message.

    :param message: The message.
    :type message: str
    :return: The message with backtick characters replaced.
    :rtype: str
    """
    return message.replace('`', '')


class ChatCompletion(object):
    """A class to call the openai chat completion endpoint."""

    def __init__(self, messages, engine, temperature,
                 max_tokens, top_p, frequency_penalty,
                 presence_penalty, stop):
        """Initialize the class.

        :param messages: The messages.
        :type messages: list
        :param engine: The engine.
        :type engine: str
        :param temperature: The temperature.
        :type temperature: float
        :param max_tokens: The maximum number of tokens.
        :type max_tokens: int
        :param top_p: The top p.
        :type top_p: float
        :param frequency_penalty: The frequency penalty.
        :type frequency_penalty: float
        :param presence_penalty: The presence penalty.
        :type presence_penalty: float
        :param stop: The stop.
        :type stop: list
        """
        self.messages = messages
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def fetch(self):
        """Call the openai chat completion endpoint.

        :return: The response.
        :rtype: dict
        """
        return openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop)


class OpenaiWrapperModel(object):
    """A model wrapper for an openai model endpoint."""

    def __init__(self, api_type, api_base, api_version, api_key,
                 engine="gpt-4-32k", temperature=0.7, max_tokens=800,
                 top_p=0.95, frequency_penalty=0, presence_penalty=0,
                 stop=None):
        """Initialize the model.

        :param api_type: The type of the API.
        :type api_type: str
        :param api_base: The base URL for the API.
        :type api_base: str
        :param api_version: The version of the API.
        :type api_version: str
        :param api_key: The API key.
        :type api_key: str
        :param engine: The engine.
        :type engine: str
        :param temperature: The temperature.
        :type temperature: float
        :param max_tokens: The maximum number of tokens.
        :type max_tokens: int
        :param top_p: The top p.
        :type top_p: float
        :param frequency_penalty: The frequency penalty.
        :type frequency_penalty: float
        :param presence_penalty: The presence penalty.
        :type presence_penalty: float
        :param stop: The stop.
        :type stop: list
        """
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def _call_webservice(self, data):
        """Common code to call the webservice.

        :param data: The data to send to the webservice.
        :type data: pandas.DataFrame or list
        :return: The result.
        :rtype: numpy.ndarray
        """
        if not rai_utils_installed:
            error = "raiutils package is required to call openai endpoint"
            raise RuntimeError(error)
        if not openai_installed:
            error = "openai package is required to call openai endpoint"
            raise RuntimeError(error)
        if not isinstance(data, list):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            else:
                data = data.values.tolist()
        openai.api_type = self.api_type
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_key = self.api_key
        answers = []
        for doc in data:
            messages = []
            messages.append({'role': 'user', CONTENT: doc})
            fetcher = ChatCompletion(messages, self.engine, self.temperature,
                                     self.max_tokens, self.top_p,
                                     self.frequency_penalty,
                                     self.presence_penalty, self.stop)
            action_name = "Call openai chat completion"
            err_msg = "Failed to call openai endpoint"
            max_retries = 4
            retry_delay = 60
            response = retry_function(fetcher.fetch, action_name, err_msg,
                                      max_retries=max_retries,
                                      retry_delay=retry_delay)
            answers.append(replace_backtick_chars(response['choices'][0]['message'][CONTENT]))
        return np.array(answers)

    def predict(self, context, model_input=None):
        """Predict using the model.

        :param context: The context for MLFlow model or the input data.
        :type context: mlflow.pyfunc.model.PythonModelContext or
            pandas.DataFrame
        :param model_input: The input to the model.
        :type model_input: pandas.DataFrame
        :return: The predictions.
        :rtype: numpy.ndarray
        """
        # This is to conform to the scikit-learn API format
        # which MLFlow does not follow
        if model_input is None:
            model_input = context
        questions = model_input['questions']
        if isinstance(questions, str):
            questions = [questions]
        result = self._call_webservice(questions)
        return result
