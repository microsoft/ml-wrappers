# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import datasets
import pandas as pd
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

EMOTION = 'emotion'


def load_emotion_dataset():
    dataset = datasets.load_dataset(EMOTION, split="train")
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
