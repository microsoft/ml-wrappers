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
from azureml.automl.dnn.vision.common.mlflow.mlflow_model_wrapper import \
    MLFlowImagesModelWrapper
from azureml.automl.dnn.vision.common.model_export_utils import (
    _get_mlflow_signature, _get_scoring_method)
from azureml.automl.dnn.vision.object_detection.common.constants import \
    ModelNames
from azureml.automl.dnn.vision.object_detection.models import \
    object_detection_model_wrappers
from common_vision_utils import load_base64_images, load_object_fridge_dataset
from ml_wrappers.common.constants import ModelTask
from wrapper_validator import validate_wrapped_object_detection_model

from ml_wrappers import wrap_model


@pytest.mark.usefixtures('_clean_dir')
class TestImageModelWrapper(object):
    # Skip for older versions of python as azureml-automl-dnn-vision
    # works with ">=3.7,<3.9"
    @pytest.mark.skipif(
        sys.version_info < (3, 7),
        reason='azureml-automl-dnn-vision not supported for older versions'
    )
    @pytest.mark.skipif(
        sys.version_info >= (3, 9),
        reason='azureml-automl-dnn-vision not supported for newer versions'
    )
    def test_wrap_automl_object_detection_model(self):
        data = load_object_fridge_dataset()[1:4]
        model_name = ModelNames.FASTER_RCNN_RESNET50_FPN

        with tempfile.TemporaryDirectory() as tmp_output_dir:

            task_type = shared_constants.Tasks.IMAGE_OBJECT_DETECTION
            number_of_classes = 4
            class_names = ['can', 'carton', 'milk_bottle', 'water_bottle']
            model_wrapper = object_detection_model_wrappers \
                .ObjectDetectionModelFactory() \
                .get_model_wrapper(number_of_classes, model_name)

            # mock for Mlflow model generation
            model_file = os.path.join(tmp_output_dir, "model.pt")
            torch.save(
                {
                    'model_name': model_name,
                    'number_of_classes': number_of_classes,
                    'model_state': copy.deepcopy(model_wrapper.state_dict()),
                    'specs': {
                        'model_settings':
                        model_wrapper.model_settings.get_settings_dict(),
                        'inference_settings': model_wrapper.inference_settings,
                        'classes': model_wrapper.classes,
                        'model_specs': {
                            'model_settings':
                            model_wrapper.model_settings.get_settings_dict(),
                            'inference_settings':
                            model_wrapper.inference_settings,
                        },
                    },
                },
                model_file
            )
            settings_file = os.path.join(
                tmp_output_dir,
                shared_constants.MLFlowLiterals.MODEL_SETTINGS_FILENAME
            )
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
                artifacts={
                    "model": model_file,
                    "settings": settings_file
                },
                conda_env=conda_env,
                signature=_get_mlflow_signature(task_type)
            )
            mlflow_model = mlflow.pyfunc.load_model(remote_path)

            # load the paths as base64 images
            data = load_base64_images(data, return_image_size=True)

            wrapped_model = wrap_model(
                mlflow_model, data, ModelTask.OBJECT_DETECTION,
                classes=class_names)
            validate_wrapped_object_detection_model(wrapped_model, data)
