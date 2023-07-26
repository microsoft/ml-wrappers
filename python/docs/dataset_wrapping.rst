.. _dataset_wrapping:

Dataset Wrapping
================

The ``DatasetWrapper`` class in the ``ml_wrappers`` package provides a uniform interface for handling datasets across different explainers. It supports various data types including numpy arrays, pandas DataFrame, pandas Series, scipy sparse matrices, shap.DenseData, torch.Tensor, and tensorflow.python.data.ops.dataset_ops.BatchDataset.

.. code-block:: python

    from ml_wrappers.dataset import DatasetWrapper

    # Initialize the dataset wrapper
    wrapper = DatasetWrapper(dataset)

Here, ``dataset`` is a matrix of feature vector examples (# examples x # features) for initializing the explainer.

The ``DatasetWrapper`` class also provides methods for operations such as summarizing data, taking the subset or sampling. It also provides an option to clear all references after use in explainers for memory optimization.

.. code-block:: python

    # Initialize the dataset wrapper with clear_references option
    wrapper = DatasetWrapper(dataset, clear_references=True)

The ``DatasetWrapper`` class also provides a method for sampling examples from the dataset. If the number of rows in the dataset is less than a lower bound, it returns the full dataset. If the number of rows is more than an upper bound, it samples randomly. It also provides an option to resample based on the optimal number of clusters.

.. code-block:: python

    # Sample examples from the dataset
    sampled_dataset = wrapper.sample_examples()

The ``DatasetWrapper`` class also provides a method to clear all references for memory optimization.

.. code-block:: python

    # Clear all references
    wrapper._clear()

The ``DatasetWrapper`` class is part of the ``ml_wrappers.dataset`` module, which also includes the ``CustomTimestampFeaturizer`` class for timestamp featurization.

.. code-block:: python

    from ml_wrappers.dataset import CustomTimestampFeaturizer

    # Initialize the timestamp featurizer
    featurizer = CustomTimestampFeaturizer()