# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import zipfile

import datasets
import pandas as pd
from raiutils.common.retries import retry_function
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


EMOTION_DATASET = 'SetFit/emotion'
EMOTION = 'emotion'
COVID19_EVENTS_MODEL_NAME = "covid19_events_model"


def load_emotion_dataset():
    dataset = datasets.load_dataset(EMOTION_DATASET, split="train")
    data = pd.DataFrame({'text': dataset['text'],
                         EMOTION: dataset['label']})
    return data


def load_squad_dataset():
    dataset = datasets.load_dataset("squad", split="train")
    answers = []
    for row in dataset['answers']:
        answers.append(row['text'][0])
    questions = []
    context = []
    for row in dataset:
        context.append(row['context'])
        questions.append(row['question'])
    data = pd.DataFrame({'context': context, 'questions': questions, 'answers': answers})
    return data


def load_covid19_emergency_event_dataset():
    dataset = datasets.load_dataset("joelito/covid19_emergency_event", split="train")
    dataset = pd.DataFrame({"language": dataset["language"],
                            "text": dataset["text"],
                            "event1": dataset["event1"],
                            "event2": dataset["event2"],
                            "event3": dataset["event3"],
                            "event4": dataset["event4"],
                            "event5": dataset["event5"],
                            "event6": dataset["event6"],
                            "event7": dataset["event7"],
                            "event8": dataset["event8"]})
    dataset = dataset[dataset.language == "en"].reset_index(drop=True)
    dataset = dataset.drop(columns="language")
    return dataset


def create_text_classification_pipeline():
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "nateraw/bert-base-uncased-emotion", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "nateraw/bert-base-uncased-emotion")

    # build a pipeline object to do predictions
    pred = pipeline("text-classification", model=model,
                    tokenizer=tokenizer, device=-1,
                    return_all_scores=True)
    return pred


def create_question_answering_pipeline():
    return pipeline('question-answering')


class FetchCovid19Model(object):
    def __init__(self):
        pass

    def fetch(self):
        zipfilename = COVID19_EVENTS_MODEL_NAME + '.zip'
        url = ('https://publictestdatasets.blob.core.windows.net/models/' +
               COVID19_EVENTS_MODEL_NAME + '.zip')
        urlretrieve(url, zipfilename)
        with zipfile.ZipFile(zipfilename, 'r') as unzip:
            unzip.extractall(COVID19_EVENTS_MODEL_NAME)


def create_multilabel_text_pipeline():
    fetcher = FetchCovid19Model()
    action_name = "Model download"
    err_msg = "Failed to download model"
    max_retries = 4
    retry_delay = 60
    retry_function(fetcher.fetch, action_name, err_msg,
                   max_retries=max_retries,
                   retry_delay=retry_delay)
    labels = ["event1", "event2", "event3", "event4", "event5", "event6", "event7", "event8"]
    num_labels = len(labels)
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        COVID19_EVENTS_MODEL_NAME, num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = -1
    # build a pipeline object to do predictions
    pred = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True
    )
    return pred
