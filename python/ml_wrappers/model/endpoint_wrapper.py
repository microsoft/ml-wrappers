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


class EndpointWrapperModel(PythonModel):
    """Defines an MLFlow model wrapper for an endpoint."""

    def __init__(self, api_key, url, allow_self_signed_https=False,
                 extra_headers=None):
        """Initialize the EndpointWrapperModel.

        :param api_key: The API key to use when invoking the endpoint.
        :type api_key: str
        :param url: The URL of the endpoint.
        :type url: str
        :param allow_self_signed_https: Whether to allow certificates.
        :type allow_self_signed_https: bool
        :param extra_headers: Extra headers to send with the request.
        :type extra_headers: dict
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

    def predict(self, context, model_input):
        """Predict using the model.

        :param context: The context.
        :type context: mlflow.pyfunc.model.PythonModelContext
        :param model_input: The input to the model.
        :type model_input: pandas.DataFrame
        :return: The predictions.
        :rtype: pandas.DataFrame
        """
        if self._allow_self_signed_https:
            # this line is needed if you use self-signed
            # certificate in your scoring service
            self.allow_self_signed_https(True)

        body = str.encode(json.dumps(model_input.values.tolist()))

        headers = {'Content-Type': 'application/json',
                   'Authorization': ('Bearer ' + self._api_key)}
        if self._extra_headers:
            headers.update(self._extra_headers)

        req = urllib.request.Request(self._url, body, headers)

        try:
            response = urllib.request.urlopen(req)

            json_data = response.read()
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the request ID
            # and the timestamp, which are useful for debugging
            # the failure
            print(error.info())
            print(error.read().decode(UTF8, IGNORE))
        try:
            result = np.array(json.loads(json_data))
        except Exception as e:
            print("Failed to convert result to numpy array, json_data:")
            print(json_data)
            print(type(json_data))
            raise ValueError(
                "Failed to convert result to numpy array. " + str(e))
        return result
