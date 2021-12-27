# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for wrapping datasets and models in one uniform format.
"""
from .dataset import DatasetWrapper
from .model import wrap_model
from .version import name, version

__all__ = ['DatasetWrapper', 'wrap_model']

import atexit
# Setup logging infrustructure
import logging
import os

# Only log to disk if environment variable specified
ml_wrappers_c_logs = os.environ.get('ML_WRAPPERS_C_LOGS')
if ml_wrappers_c_logs is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(ml_wrappers_c_logs), exist_ok=True)
    handler = logging.FileHandler(ml_wrappers_c_logs, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Initializing logging file for ml-wrappers')

    def close_handler():
        handler.close()
        logger.removeHandler(handler)
    atexit.register(close_handler)

__name__ = name
__version__ = version
