.. _pytorch_model_wrapping:

PyTorch Model Wrapping
======================

Install PyTorch separately; the image path also uses torchvision.
``wrap_model`` recognizes ``torch.nn.Module`` for tabular models and
uses ``WrappedPytorchModel`` to convert inputs and expose prediction methods.
Use CPU tensors/models for this tabular adapter: its output conversion
calls ``Tensor.numpy()`` directly.

For example, the following complete, untrained demonstration produces
two-class probability output. Replace it with a fitted model for real use:

.. code-block:: python

    import numpy as np
    import torch
    from ml_wrappers import wrap_model

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Softmax(dim=1),
    ).cpu()
    features = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    wrapped = wrap_model(model, features, model_task='classification')
    probabilities = wrapped.predict_proba(features)
    labels = wrapped.predict(features)

The adapter calls ``model.eval()`` and runs inference under
``torch.no_grad()``. ``predict_proba`` returns the model's output; it does
not add softmax or calibrate logits. Supply probabilities for classification
or request ``'regression'`` for a suitable single-target numeric model.

``WrappedPytorchModel`` also exposes ``predict_classes``. Prefer the public
``wrap_model`` entry point for task/output normalization. Initializer and
validation helpers in the repository's tests are not library APIs.

See :ref:`image_model_wrapping` for image conversion and
:ref:`object_detection_model_wrapping` for detection, which uses a
different adapter.
