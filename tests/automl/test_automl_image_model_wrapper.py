# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for wrap_model function on vision-based models"""

import copy
import json
import os
import sys
import tempfile

import azureml.automl.core.shared.constants as shared_constants
import mlflow
import pytest
import torch
from azureml.automl.dnn.vision.classification.common.constants import \
    ModelNames
from azureml.automl.dnn.vision.classification.models import ModelFactory
from azureml.automl.dnn.vision.common.mlflow.mlflow_model_wrapper import \
    MLFlowImagesModelWrapper
from azureml.automl.dnn.vision.common.model_export_utils import (
    _get_mlflow_signature, _get_scoring_method)
from common_vision_utils import load_base64_images, load_fridge_dataset
from ml_wrappers import wrap_model
from ml_wrappers.common.constants import ModelTask
from wrapper_validator import validate_wrapped_classification_model


@pytest.mark.usefixtures('_clean_dir')
class TestImageModelWrapper(object):
    # Skip for older versions of python as azureml-automl-dnn-vision
    # works with ">=3.7,<3.9"
    @pytest.mark.skipif(
        sys.version_info < (3, 7),
        reason=('azureml-automl-dnn-vision not supported '
                'for newer versions of python'))
    @pytest.mark.skipif(
        sys.version_info >= (3, 9),
        reason=('azureml-automl-dnn-vision not supported '
                'for newer versions of python'))
    def test_wrap_automl_image_classification_model(self):
        data = load_fridge_dataset()
        model_name = ModelNames.SERESNEXT
        multilabel = False
        with tempfile.TemporaryDirectory() as tmp_output_dir:

            task_type = shared_constants.Tasks.IMAGE_CLASSIFICATION
            number_of_classes = 10
            model_wrapper = ModelFactory().get_model_wrapper(
                model_name,
                number_of_classes,
                multilabel=multilabel,
                device="cpu",
                distributed=False,
                local_rank=0)

            # mock for Mlflow model generation
            model_file = os.path.join(tmp_output_dir, "model.pt")
            torch.save({
                'model_name': model_name,
                'number_of_classes': number_of_classes,
                'model_state': copy.deepcopy(model_wrapper.state_dict()),
                'specs': {
                    'multilabel': model_wrapper.multilabel,
                    'model_settings': model_wrapper.model_settings,
                    'labels': model_wrapper.labels
                },

            }, model_file)
            settings_file = os.path.join(
                tmp_output_dir,
                shared_constants.MLFlowLiterals.MODEL_SETTINGS_FILENAME)
            remote_path = os.path.join(tmp_output_dir, "outputs")

            with open(settings_file, 'w') as f:
                json.dump({}, f)

            conda_env = {
                'channels': ['conda-forge', 'pytorch'],
                'dependencies': [
                    'python=3.7',
                    'numpy==1.21.6',
                    'pytorch==1.7.1',
                    'torchvision==0.12.0',
                    {'pip': ['azureml-automl-dnn-vision']}
                ],
                'name': 'azureml-automl-dnn-vision-env'
            }

            mlflow_model_wrapper = MLFlowImagesModelWrapper(
                model_settings={},
                task_type=task_type,
                scoring_method=_get_scoring_method(task_type)
            )
            print("Saving mlflow model at {}".format(remote_path))
            mlflow.pyfunc.save_model(
                path=remote_path,
                python_model=mlflow_model_wrapper,
                artifacts={"model": model_file,
                           "settings": settings_file},
                conda_env=conda_env,
                signature=_get_mlflow_signature(task_type))
            mlflow_model = mlflow.pyfunc.load_model(remote_path)

            # load the paths as base64 images
            data = load_base64_images(data)
            wrapped_model = wrap_model(
                mlflow_model, data, ModelTask.IMAGE_CLASSIFICATION)
            validate_wrapped_classification_model(wrapped_model, data)
