# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a model wrapper for an endpoint."""

import json
import logging
import os
import ssl
import urllib.request

import numpy as np

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

ERR_MSG = 'Could not import mlflow, required if using EndpointWrapperModel'

try:
    from mlflow.pyfunc import PythonModel
    mlflow_installed = True
except ImportError:
    PythonModel = object
    mlflow_installed = False
    module_logger.debug(ERR_MSG)


UTF8 = 'utf8'
IGNORE = 'ignore'
LABEL = 'label'
SCORE = 'score'
# 6 minutes timeout for requests
TIMEOUT = 60 * 6
DEFAULT_BATCH_SIZE = 10


class EndpointWrapperModel(PythonModel):
    """Defines an MLFlow model wrapper for an endpoint."""

    def __init__(self, api_key, url, allow_self_signed_https=False,
                 extra_headers=None, transform_output_dict=False,
                 class_names=None, wrap_input_data_dict=False,
                 batch_size=DEFAULT_BATCH_SIZE,
                 api_key_auto_refresh_callable=None):
        """Initialize the EndpointWrapperModel.

        :param api_key: The API key to use when invoking the endpoint.
        :type api_key: str
        :param url: The URL of the endpoint.
        :type url: str
        :param allow_self_signed_https: Whether to allow certificates.
        :type allow_self_signed_https: bool
        :param extra_headers: Extra headers to send with the request.
        :type extra_headers: dict
        :param transform_output_dict: Whether to transform the output
            dictionary to a numpy array.
        :type transform_output_dict: bool
        :param class_names: The class names.
        :type class_names: list
        :param wrap_input_data_dict: Whether to wrap the input data in a
            dictionary format.
        :type wrap_input_data_dict: bool
        :param batch_size: The batch size to use when invoking the endpoint.
            Calls on full dataset if less than 1.
            This parameter can be used to get around endpoint timeouts.
        :type batch_size: int
        :param api_key_auto_refresh_callable: The method to call to refresh the
            API key.
        :type api_key_auto_refresh_callable: callable
        """
        if not mlflow_installed:
            raise ImportError(ERR_MSG)
        self._api_key = api_key
        if not api_key:
            key_err = "A key should be provided to invoke the endpoint"
            raise ValueError(key_err)
        self._url = url
        self._allow_self_signed_https = allow_self_signed_https
        self._extra_headers = extra_headers
        self._model = self
        self._transform_output_dict = transform_output_dict
        self._class_names = class_names
        self._wrap_input_data_dict = wrap_input_data_dict
        self._batch_size = batch_size
        self._api_key_auto_refresh_callable = api_key_auto_refresh_callable

    @staticmethod
    def from_auto_refresh_callable(api_key_auto_refresh_callable, url, **kwargs):
        """Create an EndpointWrapperModel from an auto refresh callable.

        The callable method should return the latest API key.

        :param api_key_auto_refresh_callable: The method to call to refresh the
            API key.
        :type api_key_auto_refresh_callable: callable
        :param kwargs: The keyword arguments.
        :type kwargs: dict
        :return: The EndpointWrapperModel.
        :rtype: ml_wrappers.model.EndpointWrapperModel
        """
        api_key = api_key_auto_refresh_callable()
        return EndpointWrapperModel(
            api_key,
            url,
            api_key_auto_refresh_callable=api_key_auto_refresh_callable,
            **kwargs)

    def load_context(self, context):
        """Load the context.

        :param context: The context.
        :type context: mlflow.pyfunc.model.PythonModelContext
        """
        # load your artifacts
        pass

    def allow_self_signed_https(self, allowed):
        """Allow self signed HTTPS.

        :param allowed: Whether to allow self signed HTTPS.
        :type allowed: bool
        """
        # bypass the server certificate verification on client side
        verify = os.environ.get('PYTHONHTTPSVERIFY', '')
        ssl_verified = getattr(ssl, '_create_unverified_context', None)
        if allowed and not verify and ssl_verified:
            ssl._create_default_https_context = ssl._create_unverified_context

    def _get_response_with_retry(self, body, timeout, num_retries=3):
        try:
            headers = {'Content-Type': 'application/json',
                       'Authorization': ('Bearer ' + self._api_key)}
            if self._extra_headers:
                headers.update(self._extra_headers)
            req = urllib.request.Request(self._url, body, headers)
            response = urllib.request.urlopen(req, timeout=TIMEOUT)
            json_data = response.read()
        except urllib.error.HTTPError as e:
            print("The request failed with status code: " + str(e.code))
            print("Request body: " + str(body))

            # Print the headers - they include the request ID
            # and the timestamp, which are useful for debugging
            # the failure
            print(e.info())
            # Retry request and refresh API key if refresh method provided
            if num_retries > 0:
                if self._api_key_auto_refresh_callable:
                    self._api_key = self._api_key_auto_refresh_callable()
                return self._get_response_with_retry(
                    body, timeout, num_retries - 1)
            raise ValueError(
                "Request failed with error. " + str(e))
        return json_data

    def _make_request(self, data):
        """Make a request to the endpoint.

        :param data: The data to send to the endpoint.
        :type data: list
        :return: The result.
        :rtype: numpy.ndarray
        """
        if self._wrap_input_data_dict:
            data = {'input_data': data}
        json_input_data = json.dumps(data)
        body = str.encode(json_input_data)
        json_data = self._get_response_with_retry(body, TIMEOUT)
        try:
            result = np.array(json.loads(json_data))
        except Exception as e:
            print("Failed to convert result to numpy array, json_data:")
            print(json_data)
            print(type(json_data))
            raise ValueError(
                "Failed to convert result to numpy array. " + str(e))
        return result

    def _call_webservice(self, data):
        """Common code to call the webservice.

        :param data: The data to send to the webservice.
        :type data: pandas.DataFrame or list
        :return: The result.
        :rtype: numpy.ndarray
        """
        if self._allow_self_signed_https:
            # this line is needed if you use self-signed
            # certificate in your scoring service
            self.allow_self_signed_https(True)

        if not isinstance(data, list):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            else:
                data = data.values.tolist()
        batch_size = self._batch_size
        lt_batch_size = isinstance(data, list) and len(data) <= batch_size
        request_all = batch_size < 1 or lt_batch_size
        if request_all:
            return self._make_request(data)
        else:
            result = []
            for i in range(0, len(data), batch_size):
                batch_result = self._make_request(data[i:i + batch_size])
                result.extend(batch_result.tolist())
            return np.array(result)

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
        result = self._call_webservice(model_input)
        if self._transform_output_dict:
            array_result = []
            for row in result:
                label = row[LABEL]
                prediction = self._class_names.index(label)
                array_result.append(prediction)
            result = np.array(array_result)
        return result

    def predict_proba(self, data):
        """Predict using the model.

        :param data: The input to the model.
        :type data: pandas.DataFrame
        :return: The predictions.
        :rtype: numpy.ndarray
        """
        result = self._call_webservice(data)
        if self._transform_output_dict:
            array_result = []
            for row in result:
                label = row[LABEL]
                score = row[SCORE]
                if label == self._class_names[0]:
                    array_result.append([score, 1 - score])
                else:
                    array_result.append([1 - score, score])
            result = np.array(array_result)
        return result

    def __call__(self, data):
        """Call the model.

        :param data: The input to the model.
        :type data: pandas.DataFrame
        :return: The predictions.
        :rtype: numpy.ndarray
        """
        return self.predict_proba(data)
