.. _supported_frameworks:

Supported Frameworks
====================

Support is scenario-specific, not a guarantee for every model in a framework.
The base package does not install optional ML frameworks. This is the
canonical support summary; see :ref:`dependencies` for installation sources
and CI version constraints.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Scenario
     - Model or framework
     - Contract and prerequisites
   * - Tabular classification/regression
     - scikit-learn, LightGBM, XGBoost, CatBoost
     - A fitted estimator with suitable ``predict``/``predict_proba`` output.
       Install non-scikit-learn frameworks separately.
   * - Tabular neural networks
     - PyTorch; recognized Keras Sequential models
     - Install PyTorch or TensorFlow/Keras. Supply model outputs appropriate
       for the task; see the framework guides.
   * - Tabular learners
     - FastAI
     - A compatible tabular learner; install FastAI and its framework stack.
   * - Text classification
     - Hugging Face Transformers pipelines
     - Transformers, a backend and SHAP; explicit ``text_classification`` or
       ``multilabel_text_classification`` task.
   * - Question answering
     - Hugging Face Transformers pipelines
     - Transformers, a backend and SHAP for public pipeline recognition;
       explicit ``question_answering`` task.
   * - Image classification
     - PyTorch, FastAI, compatible Transformers/callable pipelines,
       AzureML AutoML Images MLflow models
     - Framework-specific input/output contracts; see
       :ref:`image_model_wrapping`. Multilabel image adaptation is
       implemented for FastAI.
   * - Object detection
     - Torchvision-style detectors; AzureML AutoML Images MLflow models
     - Automatic wrapping of compatible callable detectors or explicit adapter
       construction, with scenario-specific schemas; see
       :ref:`object_detection_model_wrapping`. Requires PyTorch/torchvision;
       the MLflow path also needs its AutoML dependencies.
   * - Remote inference
     - Configured endpoints; OpenAI/Azure OpenAI clients
     - Explicit wrappers, service access and optional dependencies.
       Inputs are sent to the configured service; see :ref:`privacy_policy`.
   * - Stored predictions
     - PredictionsModelWrapper family
     - Matching recorded rows and predictions; not inference on arbitrary
       new or perturbed inputs.

ONNX is not currently supported. Other text task names in ``ModelTask`` do
not imply an implemented pipeline adapter. A generic callable or arbitrary
MLflow model is not automatically supported merely because a related
scenario appears here.
