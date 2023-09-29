# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the EndpointWrapperModel class."""

import json
import urllib.request
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ml_wrappers.model import EndpointWrapperModel


class MockRead():
    """Mock class for urllib.request.urlopen().read()"""

    def __init__(self, json_data, fail_read=False):
        """Initialize the MockRead class.

        :param json_data: The json data to return from the read method.
        :type json_data: str
        """
        self.json_data = json_data
        self.fail_read = fail_read

    def read(self):
        """Return the json data.

        :return: The json data.
        :rtype: str
        """
        if self.fail_read:
            # reset fail_read to False so that the next call to read
            # does not fail
            self.fail_read = False
            raise urllib.error.HTTPError('url', 500, 'Internal Server Error', {}, None)
        return self.json_data


def mock_api_key_auto_refresh_method():
    """Mock method for auto refreshing the API key.

    :return: The mock API key.
    :rtype: str
    """
    return 'mock_key'


@pytest.mark.usefixtures('_clean_dir')
class TestEndpointWrapperModel(object):
    def test_predict_call(self):
        # test creating the EndpointWrapperModel and
        # calling the predict function
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        endpoint_wrapper = EndpointWrapperModel('mock_key', 'http://mock.url')
        # mock the urllib.request.urlopen function
        with patch('urllib.request.urlopen') as mock_urlopen:
            json_inference_value = json.dumps(test_dataframe.values.tolist())
            # wrap return value in mock class with read method
            mock_urlopen.return_value = MockRead(json_inference_value)
            context = {}
            result = endpoint_wrapper.predict(context, test_dataframe)
            # assert result and test_dataframe.values equal
            assert np.array_equal(result, test_dataframe.values)

    def test_auto_refresh_token(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        endpoint_wrapper = EndpointWrapperModel.from_auto_refresh_callable(
            mock_api_key_auto_refresh_method,
            'http://mock.url')
        # mock the urllib.request.urlopen function
        with patch('urllib.request.urlopen') as mock_urlopen:
            json_inference_value = json.dumps(test_dataframe.values.tolist())
            # wrap return value in mock class with read method
            mock_urlopen.return_value = MockRead(
                json_inference_value, fail_read=True)
            context = {}
            result = endpoint_wrapper.predict(context, test_dataframe)
            # assert result and test_dataframe.values equal
            assert np.array_equal(result, test_dataframe.values)
