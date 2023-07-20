.. _text_model_wrapping:

Text Model Wrapping
===================

The ml-wrappers library provides support for wrapping text-based models. This includes both classification and question-answering models. 

WrappedQuestionAnsweringModel
-----------------------------

The WrappedQuestionAnsweringModel class is used for wrapping a Transformers model in the scikit-learn style. 

.. code-block:: python

    class WrappedQuestionAnsweringModel(object):
        """A class for wrapping a Transformers model in the scikit-learn style."""

        def __init__(self, model):
            """Initialize the WrappedQuestionAnsweringModel."""
            self._model = model

        def predict(self, dataset):
            """Predict the output using the wrapped Transformers model.

            :param dataset: The dataset to predict on.
            :type dataset: ml_wrappers.DatasetWrapper
            """
            output = []
            for context, question in zip(dataset['context'], dataset['questions']):
                answer = self._model({'context': context, 'question': question})
                output.append(answer['answer'])
            return output

WrappedTextClassificationModel
------------------------------

The WrappedTextClassificationModel class is used for wrapping a Transformers model in the scikit-learn style. 

.. code-block:: python

    class WrappedTextClassificationModel(object):
        """A class for wrapping a Transformers model in the scikit-learn style."""

        def __init__(self, model, multilabel=False):
            """Initialize the WrappedTextClassificationModel."""
            self._model = model
            if not shap_installed:
                raise ImportError("SHAP is not installed. Please install it " +
                                  "to use WrappedTextClassificationModel.")
            self._wrapped_model = models.TransformersPipeline(model)
            self._multilabel = multilabel

        def predict(self, dataset):
            """Predict the output using the wrapped Transformers model.

            :param dataset: The dataset to predict on.
            :type dataset: ml_wrappers.DatasetWrapper
            """
            pipeline_dicts = self._wrapped_model.inner_model(dataset)
            output = []
            for val in pipeline_dicts:
                if not isinstance(val, list):
                    val = [val]
                scores = [obj["score"] for obj in val]
                if self._multilabel:
                    threshold = MULTILABEL_THRESHOLD
                    labels = np.where(np.array(scores) > threshold)
                    predictions = np.zeros(len(scores))
                    predictions[labels] = 1
                    output.append(predictions)
                else:
                    max_score_index = np.argmax(scores)
                    output.append(max_score_index)
            return np.array(output)

        def predict_proba(self, dataset):
            """Predict the output probability using the Transformers model.

            :param dataset: The dataset to predict_proba on.
            :type dataset: ml_wrappers.DatasetWrapper
            """
            return self._wrapped_model(dataset)

The wrap_model function is used to wrap the model. It takes as input the model, the data, and the model task (in this case, text classification or question answering). The function returns a wrapped model that can be used for further processing or evaluation. 

.. code-block:: python

    from ml_wrappers import wrap_model
    from ml_wrappers.common.constants import ModelTask

    wrapped_model = wrap_model(model, data, ModelTask.TEXT_CLASSIFICATION)

For more information on how to use these classes and functions, please refer to the source code and the provided examples.