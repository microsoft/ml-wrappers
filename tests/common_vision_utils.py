# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import base64
import json
import os
import sys
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shap
from PIL import Image

try:
    from fastai.data.transforms import Normalize
    from fastai.learner import load_learner
    from fastai.losses import BCEWithLogitsLossFlat
    from fastai.metrics import accuracy, accuracy_multi
    from fastai.vision import models
    from fastai.vision.augment import Resize
    from fastai.vision.data import ImageDataLoaders, imagenet_stats
    from fastai.vision.learner import vision_learner
except (ImportError, SyntaxError):
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
MULTILABEL_FRIDGE_MODEL_NAME = 'multilabel_fridge_model'
MULTILABEL_FRIDGE_MODEL_WINDOWS_NAME = 'multilabel_fridge_model_windows'
WIN = 'win'


def load_imagenet_dataset():
    X, _ = shap.datasets.imagenet50()
    # load just the first 10 images
    X = X[:10]
    return X


def load_imagenet_labels():
    # getting ImageNet 1000 class names
    url = "https://s3.amazonaws.com/deep-learning-models/image-models"
    url_ending = "/imagenet_class_index.json"
    with open(shap.datasets.cache(url + url_ending)) as file:
        class_names = [v[1] for v in json.load(file).values()]
    return class_names


def retrieve_unzip_file(download_url, data_file):
    urlretrieve(download_url, filename=data_file)
    # extract files
    with ZipFile(data_file, "r") as zipfile:
        zipfile.extractall(path="./data")
    # delete zip file
    os.remove(data_file)


def load_fridge_dataset():
    # create data folder if it doesnt exist.
    os.makedirs("data", exist_ok=True)

    # download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/"
    download_url_end = "image_classification/fridgeObjects.zip"
    data_file = "./data/fridgeObjects.zip"
    retrieve_unzip_file(download_url + download_url_end, data_file)

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


def load_multilabel_fridge_dataset():
    # create data folder if it doesnt exist.
    os.makedirs("data", exist_ok=True)

    # download data
    download_url = ("https://cvbp-secondary.z19.web.core.windows.net/"
                    "datasets/image_classification/"
                    "multilabelFridgeObjects.zip")
    folder_path = './data/multilabelFridgeObjects'
    data_file = folder_path + '.zip'
    retrieve_unzip_file(download_url, data_file)

    data = pd.read_csv(folder_path + '/labels.csv')
    data.rename(columns={'filename': IMAGE,
                         'labels': LABEL}, inplace=True)
    image_col = data[IMAGE]
    for i in range(len(image_col)):
        image_col[i] = folder_path + '/images/' + image_col[i]
    return data


def load_images(data):
    images = []
    for image_path in data[IMAGE]:
        with Image.open(image_path) as im:
            image = np.array(im)
        images.append(image)
    return np.array(images)


def get_base64_string_from_path(img_path: str,
                                return_image_size: bool = False) \
        -> Union[str, Tuple[str, Tuple[int, int]]]:
    """Load and convert pillow image to base64-encoded image

    :param img_path: image path
    :type img_path: str
    :param return_image_size: true if image sizes should be returned in
                            dataframe
    :type data: bool
    :return: base64-encoded image OR a tuple containing a base64-encoded image
            and a tuple with the image size
    :rtype: Union[str, Tuple[str, Tuple[int, int]]]
    """
    img = Image.open(img_path)
    imgio = BytesIO()
    img.save(imgio, img.format)
    img_str = base64.b64encode(imgio.getvalue())
    decoded_img = img_str.decode("utf-8")
    if return_image_size:
        return decoded_img, img.size
    return decoded_img


def load_base64_images(data: pd.DataFrame, return_image_size: bool = False) \
        -> pd.DataFrame:
    """Create dataframe of images encoded in base64 format (and optionally
        their sizes)

    :param data: input data with image paths and lables
    :type data: pandas.DataFrame
    :param return_image_size: true if image sizes should be returned in
                              dataframe
    :type data: bool
    :return: base64-encoded image
    :rtype: pandas.DataFrame
    """
    if return_image_size:
        dataset = pd.DataFrame(
            data=[[x for x in get_base64_string_from_path(
                img_path, return_image_size=True)] for img_path in
                data.loc[:, IMAGE]],
            columns=[IMAGE, "image_size"],
        )
        return dataset
    data.loc[:, IMAGE] = data.loc[:, IMAGE].map(
        lambda img_path: get_base64_string_from_path(img_path))
    return data.loc[:, [IMAGE]]


class ResNetPipeline(object):
    def __init__(self):
        self.model = ResNet50(weights='imagenet')

    def __call__(self, X):
        tmp = X.copy()
        preprocess_input(tmp)
        return self.model(tmp)


def create_image_classification_pipeline():
    """Create a pipeline for image classification.

    :return: pipeline
    :rtype: ResNetPipeline
    """
    return ResNetPipeline()


class ScikitResNetPipeline(object):
    def __init__(self):
        """Creates a scikit-learn compatible pipeline for image classification.
        """
        self.model = ResNet50(weights='imagenet')

    def predict(self, X):
        """Predicts the class for each image in the dataset.

        :param X: dataset
        :type X: numpy.ndarray
        :return: predicted classes
        :rtype: numpy.ndarray
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Predicts the probability of each class for each image in the dataset.

        :param X: dataset
        :type X: numpy.ndarray
        :return: predicted probabilities
        :rtype: numpy.ndarray
        """
        tmp = X.copy()
        preprocess_input(tmp)
        return self.model(tmp)


def create_scikit_classification_pipeline():
    """Create a scikit-learn compatible pipeline for image classification.

    :return: scikit-learn compatible pipeline
    :rtype: ScikitResNetPipeline
    """
    return ScikitResNetPipeline()


class FetchModel(object):
    def __init__(self, multilabel=False):
        self.multilabel = multilabel

    def fetch(self):
        if sys.platform.startswith(WIN):
            if self.multilabel:
                model_name = MULTILABEL_FRIDGE_MODEL_WINDOWS_NAME
            else:
                model_name = FRIDGE_MODEL_WINDOWS_NAME
        else:
            if self.multilabel:
                model_name = MULTILABEL_FRIDGE_MODEL_NAME
            else:
                model_name = FRIDGE_MODEL_NAME
        url = ('https://publictestdatasets.blob.core.windows.net/models/' + model_name)
        saved_model_name = FRIDGE_MODEL_NAME
        if self.multilabel:
            saved_model_name = MULTILABEL_FRIDGE_MODEL_NAME
        urlretrieve(url, saved_model_name)


def train_fastai_image_classifier(df):
    """Trains a fastai multiclass image classifier.

    :param df: dataframe with image paths and labels
    :type df: pandas.DataFrame
    :return: fastai vision learner
    :rtype: fastai.vision.learner
    """
    data = ImageDataLoaders.from_df(
        df, valid_pct=0.2, seed=10, bs=BATCH_SIZE,
        batch_tfms=[Resize(IM_SIZE), Normalize.from_stats(*imagenet_stats)])
    model = vision_learner(data, models.resnet18, metrics=[accuracy])
    model.unfreeze()
    model.fit(EPOCHS, LEARNING_RATE)
    return model


def train_fastai_image_multilabel(df):
    """Trains fastai image classifier for multilabel classification

    :param df: dataframe with image paths and labels
    :type df: pandas.DataFrame
    :return: trained fastai model
    :rtype: fastai.vision.learner.Learner
    """
    data = ImageDataLoaders.from_df(
        df, valid_pct=0.2, seed=10, label_delim=' ', bs=BATCH_SIZE,
        batch_tfms=[Resize(IM_SIZE), Normalize.from_stats(*imagenet_stats)])
    model = vision_learner(data, models.resnet18,
                           metrics=[accuracy_multi],
                           loss_func=BCEWithLogitsLossFlat())
    model.unfreeze()
    model.fit(EPOCHS, LEARNING_RATE)
    return model


def retrieve_or_train_fridge_model(df, force_train=False,
                                   multilabel=False):
    """Retrieves or trains fastai image classifier

    :param df: dataframe with image paths and labels
    :type df: pandas.DataFrame
    :param force_train: whether to force training of model
    :type force_train: bool
    :param multilabel: whether to train multilabel classifier
    :type multilabel: bool
    """
    model_name = FRIDGE_MODEL_NAME
    if multilabel:
        model_name = MULTILABEL_FRIDGE_MODEL_NAME
    if force_train:
        if multilabel:
            model = train_fastai_image_multilabel(df)
        else:
            model = train_fastai_image_classifier(df)
        # Save model to disk
        model.export(model_name)
    else:
        fetcher = FetchModel(multilabel)
        action_name = "Fridge model download"
        err_msg = "Failed to download model"
        max_retries = 4
        retry_delay = 60
        retry_function(fetcher.fetch, action_name, err_msg,
                       max_retries=max_retries,
                       retry_delay=retry_delay)
        model = load_learner(model_name)
    return model


def create_pytorch_image_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model


def preprocess_imagenet_dataset(dataset):
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess(Tensor(np.moveaxis(dataset, -1, 0)))


def load_object_fridge_dataset_labels():

    src_images = "./data/odFridgeObjects/"

    # Path to the annotations
    annotations_folder = os.path.join(src_images, "annotations")

    labels = []

    # Read each annotation
    for _, filename in enumerate(os.listdir(annotations_folder)):
        if filename.endswith(".xml"):
            print("Parsing " + os.path.join(src_images, filename))

            root = ET.parse(os.path.join(annotations_folder,
                                         filename)).getroot()

            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            image_labels = []
            for o in root.findall("object"):
                name = o.find("name").text
                xmin = o.find("bndbox/xmin").text
                ymin = o.find("bndbox/ymin").text
                xmax = o.find("bndbox/xmax").text
                ymax = o.find("bndbox/ymax").text
                isCrowd = int(o.find("difficult").text)
                image_labels.append(
                    {
                        "label": name,
                        "topX": float(xmin) / width,
                        "topY": float(ymin) / height,
                        "bottomX": float(xmax) / width,
                        "bottomY": float(ymax) / height,
                        "isCrowd": isCrowd,
                    }
                )
            labels.append(image_labels)
    return labels


def load_object_fridge_dataset():
    # create data folder if it doesnt exist.
    os.makedirs("data", exist_ok=True)

    # download data
    download_url = ("https://cvbp-secondary.z19.web.core.windows.net/"
                    "datasets/object_detection/odFridgeObjects.zip")
    data_file = "./odFridgeObjects.zip"
    urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as z:
        print("extracting files...")
        z.extractall(path="./data")
        print("done")
    os.remove(data_file)

    labels = load_object_fridge_dataset_labels()

    # get all file names into a pandas dataframe with the labels
    data = pd.DataFrame(columns=["image", "label"])
    for i, file in enumerate(os.listdir("./data/odFridgeObjects/" + "images")):
        image_path = "./data/odFridgeObjects/" + "images" + "/" + file
        data = data.append({"image": image_path,
                            "label": labels[i]}, ignore_index=True)
    return data
