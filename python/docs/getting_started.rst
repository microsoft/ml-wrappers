.. _getting_started:

Getting Started
===============

This documentation provides an overview of the ML Wrappers SDK, which is designed to provide a uniform format for wrapping datasets and models. 

Installation
------------

The ML Wrappers SDK can be installed via pip:

.. code-block:: bash

    pip install ml-wrappers

Supported Models
----------------

The ML Wrappers SDK supports the following models:

- Scikit-Learn
- LightGBM
- XGBoost
- Catboost
- Keras with Tensorflow backend
- Pytorch
- ONNX (planned for future support)

For more details, please refer to the :ref:`supported_models` section.

Supported Frameworks
--------------------

The ML Wrappers SDK supports the following frameworks:

- Scikit-Learn
- LightGBM
- XGBoost
- Catboost
- Keras with Tensorflow backend
- Pytorch
- ONNX (planned for future support)

For more details, please refer to the :ref:`supported_frameworks` section.

Model Wrapping
--------------

The ML Wrappers SDK provides a way to wrap models into a uniform format. This is done by either using the predict_proba function, or, if it is not available, the predict function. For more details, please refer to the :ref:`model_wrapping` section.

Dataset Wrapping
----------------

The ML Wrappers SDK provides a way to wrap datasets into a uniform format. This is done using the DatasetWrapper class. For more details, please refer to the :ref:`dataset_wrapping` section.

License Information
-------------------

The ML Wrappers SDK is licensed under the MIT License. For more details, please refer to the :ref:`license_information` section.

Support
-------

Support for this project is limited to the resources listed in the :ref:`support` section.