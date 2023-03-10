# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the EndpointWrapperModel class."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ml_wrappers.model import EndpointWrapperModel


class MockRead():
    """Mock class for urllib.request.urlopen().read()"""

    def __init__(self, json_data):
        """Initialize the MockRead class.

        :param json_data: The json data to return from the read method.
        :type json_data: str
        """
        self.json_data = json_data

    def read(self):
        """Return the json data.

        :return: The json data.
        :rtype: str
        """
        return self.json_data


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
