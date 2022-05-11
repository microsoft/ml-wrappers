# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import json

import shap
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def load_imagenet_dataset():
    X, _ = shap.datasets.imagenet50()
    # load just the first 10 images
    X = X[:10]
    return X


def load_imagenet_labels():
    # getting ImageNet 1000 class names
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with open(shap.datasets.cache(url)) as file:
        class_names = [v[1] for v in json.load(file).values()]
    return class_names


class ResNetPipeline(object):
    def __init__(self):
        self.model = ResNet50(weights='imagenet')

    def __call__(self, X):
        tmp = X.copy()
        preprocess_input(tmp)
        return self.model(tmp)


def create_image_classification_pipeline():
    return ResNetPipeline()
