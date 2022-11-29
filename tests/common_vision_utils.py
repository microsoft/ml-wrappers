# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import base64
import json
import os
import sys
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shap
from PIL import Image

try:
    from fastai.data.transforms import Normalize
    from fastai.learner import load_learner
    from fastai.metrics import accuracy
    from fastai.vision import models
    from fastai.vision.augment import Resize
    from fastai.vision.data import ImageDataLoaders, imagenet_stats
    from fastai.vision.learner import vision_learner
except SyntaxError:
    # Skip for older versions of python due to breaking changes in fastai
    pass
from raiutils.common.retries import retry_function
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from torch import Tensor

try:
    from torchvision.models import ResNet50_Weights, resnet50
except ImportError:
    # Skip for older versions of python due to recent torchvision updates
    pass

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


EPOCHS = 10
LEARNING_RATE = 1e-4
IM_SIZE = 300
BATCH_SIZE = 16
IMAGE = 'image'
LABEL = 'label'
FRIDGE_MODEL_NAME = 'fridge_model'
FRIDGE_MODEL_WINDOWS_NAME = 'fridge_model_windows'
WIN = 'win'


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


def load_fridge_dataset():
    # create data folder if it doesnt exist.
    os.makedirs("data", exist_ok=True)

    # download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip"
    data_file = "./data/fridgeObjects.zip"
    urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zipfile:
        zipfile.extractall(path="./data")
    # delete zip file
    os.remove(data_file)
    # get all file names into a pandas dataframe with the labels
    data = pd.DataFrame(columns=[IMAGE,
                                 LABEL])
    for folder in os.listdir("./data/fridgeObjects"):
        for file in os.listdir("./data/fridgeObjects/" + folder):
            image_path = "./data/fridgeObjects/" + folder + "/" + file
            data = data.append({IMAGE: image_path,
                                LABEL: folder},
                               ignore_index=True)
    return data


def load_images(data):
    images = []
    for image_path in data[IMAGE]:
        with Image.open(image_path) as im:
            image = np.array(im)
        images.append(image)
    return np.array(images)


def get_base64_string_from_path(img_path):
    """Load and convert pillow image to base64-encoded image

    :param img_path: image path
    :type img_path: str
    :return: base64-encoded image
    :rtype: str
    """
    img = Image.open(img_path)
    imgio = BytesIO()
    img.save(imgio, img.format)
    img_str = base64.b64encode(imgio.getvalue())
    return img_str.decode("utf-8")


def load_images_for_automl_images(data):
    """Create dataframe of images encoded in base64 format

    :param data: input data with image paths and lables
    :type data: pandas.DataFrame
    :return: base64-encoded image
    :rtype: pandas.DataFrame
    """
    data.loc[:, IMAGE] = data.loc[:, IMAGE].map(lambda img_path: get_base64_string_from_path(img_path))
    return data.loc[:, [IMAGE]]


class ResNetPipeline(object):
    def __init__(self):
        self.model = ResNet50(weights='imagenet')

    def __call__(self, X):
        tmp = X.copy()
        preprocess_input(tmp)
        return self.model(tmp)


def create_image_classification_pipeline():
    return ResNetPipeline()


class FetchModel(object):
    def __init__(self):
        pass

    def fetch(self):
        if sys.platform.startswith(WIN):
            model_name = FRIDGE_MODEL_WINDOWS_NAME
        else:
            model_name = FRIDGE_MODEL_NAME
        url = ('https://publictestdatasets.blob.core.windows.net/models/' +
               model_name)
        urlretrieve(url, FRIDGE_MODEL_NAME)


def train_fastai_image_classifier(df):
    data = ImageDataLoaders.from_df(
        df, valid_pct=0.2, seed=10, bs=BATCH_SIZE,
        batch_tfms=[Resize(IM_SIZE), Normalize.from_stats(*imagenet_stats)])
    model = vision_learner(data, models.resnet18, metrics=[accuracy])
    model.unfreeze()
    model.fit(EPOCHS, LEARNING_RATE)
    return model


def retrieve_or_train_fridge_model(df, force_train=False):
    if force_train:
        model = train_fastai_image_classifier(df)
        # Save model to disk
        model.export(FRIDGE_MODEL_NAME)
    else:
        fetcher = FetchModel()
        action_name = "Dataset download"
        err_msg = "Failed to download dataset"
        max_retries = 4
        retry_delay = 60
        retry_function(fetcher.fetch, action_name, err_msg,
                       max_retries=max_retries,
                       retry_delay=retry_delay)
        model = load_learner(FRIDGE_MODEL_NAME)
    return model


def create_pytorch_image_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model


def preprocess_imagenet_dataset(dataset):
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess(Tensor(np.moveaxis(dataset, -1, 0)))
