# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests minimal imports and functions from ml-wrappers"""

import pytest


@pytest.mark.usefixtures('_clean_dir')
class TestMinialImports(object):
    def test_main_import(self):
        import ml_wrappers  # noqa

    def test_import_wrap_model(self):
        from ml_wrappers import wrap_model  # noqa

    def test_import_constants(self):
        from ml_wrappers.common.constants import ModelTask  # noqa
