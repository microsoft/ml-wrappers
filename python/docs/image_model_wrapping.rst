.. _image_model_wrapping:

Image Model Wrapping
====================

The ML-Wrappers SDK supports model wrapping for vision-based models. The wrapping process is handled by the ``wrap_model`` function, which takes in a model, data, and a model task as parameters. The model task can be one of the following: ``ModelTask.IMAGE_CLASSIFICATION``, ``ModelTask.MULTILABEL_IMAGE_CLASSIFICATION``, or ``ModelTask.OBJECT_DETECTION``.

The ``wrap_model`` function determines the type of the model and wraps it accordingly. For instance, if the model is a FastAI model, it is wrapped as a ``WrappedFastAIImageClassificationModel``. If the model is an AutoML model, it is wrapped as a ``WrappedMlflowAutomlImagesClassificationModel`` or a ``WrappedMlflowAutomlObjectDetectionModel`` depending on the model task. If the model is a callable pipeline, it is wrapped as a ``WrappedTransformerImageClassificationModel``.

For object detection models, the ``wrap_model`` function can also take in an additional parameter, ``classes``, which is a list of class labels. The function returns the wrapped model and the model task.

The wrapped model can then be used for various tasks such as validation and prediction. For instance, the ``validate_wrapped_classification_model`` function can be used to validate a wrapped classification model.

The ML-Wrappers SDK also provides support for PyTorch models. The ``PytorchDRiseWrapper`` and ``WrappedObjectDetectionModel`` classes are used to wrap PyTorch models for object detection tasks.

.. note::
   The ML-Wrappers SDK currently only supports PyTorch machine learning models for object detection tasks.

For more information on how to use the ML-Wrappers SDK for image model wrapping, refer to the `tests/main/test_image_model_wrapper.py` file in the repository.