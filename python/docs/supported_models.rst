.. _supported_models:

Supported Models and Adapters
=============================

For the scenario/framework matrix and prerequisites, see
:ref:`supported_frameworks`. The main entry point is
``ml_wrappers.wrap_model``; explicit wrappers serve specialized use cases.

* :ref:`model_wrapping`: fitted tabular classifiers and regressors, including
  :ref:`pytorch_model_wrapping` and :ref:`tensorflow_model_wrapping`.
* :ref:`text_model_wrapping`: supported Transformers classification and
  question-answering pipelines; public dispatch requires SHAP for both.
* :ref:`image_model_wrapping`: framework-specific image classification.
* :ref:`object_detection_model_wrapping`: detection records and D-RISE
  adapters, with distinct model/input contracts.

Stored predictions
-------------------

``PredictionsModelWrapper``, ``PredictionsModelWrapperClassification`` and
``PredictionsModelWrapperRegression`` expose recorded predictions when
loading the original model is not possible. They match requested rows to
recorded data; they cannot predict genuinely new feature combinations.
See their constructors in :ref:`API Reference` for required arrays and
classification probabilities.

Remote models
--------------

``EndpointWrapperModel`` and ``OpenaiWrapperModel`` are explicit adapters
for supported service contracts, not automatic ``wrap_model`` dispatch
for arbitrary remote URLs. Install their optional dependencies and
configure service access separately. Consult their API signatures and
:ref:`privacy_policy`; do not put credentials in examples or issue reports.
