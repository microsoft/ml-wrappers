.. _text_model_wrapping:

Text Model Wrapping
===================

Use a compatible Hugging Face Transformers pipeline and specify its task
explicitly. All public Transformers text dispatch, including question
answering, requires SHAP for pipeline recognition in addition to Transformers
and a model backend. Without SHAP, ``wrap_model`` can return the original
pipeline unchanged, which does not provide the wrapper's ``predict`` API.
Model/tokenizer downloads may require network access; load
an approved local model when working offline.

Classification
---------------

Given an already loaded, compatible text-classification pipeline named
``pipeline``, configured to return all class scores:

.. code-block:: python

    from ml_wrappers import wrap_model

    texts = ['A useful explanation.', 'This example is unclear.']
    wrapped = wrap_model(pipeline, texts, model_task='text_classification')
    labels = wrapped.predict(texts)
    scores = wrapped.predict_proba(texts)

``predict`` returns numeric positions in the pipeline's returned score list,
not the original label strings. Return every class in a consistent order;
top-label-only results are insufficient for a full class-probability matrix.
``predict_proba`` delegates to SHAP's Transformers adapter, whose output
representation can depend on that adapter's version and configuration.
Do not assume it always returns normalized probabilities.

For multilabel classification, use
``model_task='multilabel_text_classification'``; ``predict`` returns an
indicator array using the adapter's score threshold of 0.5.

Question answering
-------------------

Given an already loaded question-answering pipeline named ``qa_pipeline``:

.. code-block:: python

    import pandas as pd
    from ml_wrappers import wrap_model

    examples = pd.DataFrame({
        'context': ['ML Wrappers adapts model prediction APIs.'],
        'questions': ['What does ML Wrappers adapt?'],
    })
    wrapped = wrap_model(
        qa_pipeline, examples, model_task='question_answering')
    answers = wrapped.predict(examples)

The input columns must be named ``context`` and ``questions``. The adapter
returns a NumPy array of answer strings; it does not provide classification
probabilities. Other text task constants, including summarization and
entailment, do not currently select an implemented text wrapper.
