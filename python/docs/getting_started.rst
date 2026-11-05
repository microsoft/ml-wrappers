.. _getting_started:

Getting Started
===============

Install the published package:

.. code-block:: bash

    python -m pip install ml-wrappers

The base package installs NumPy, packaging, pandas, SciPy and scikit-learn.
Other frameworks are optional: install the dependencies for your scenario
before importing or using its adapter. See :ref:`dependencies` and
:ref:`supported_frameworks`. There are no framework-specific package extras.

Wrap a classifier
-----------------

This self-contained example needs only the base package. ``wrap_model``
evaluates the supplied examples while choosing a prediction contract, so use
representative, valid model inputs and a fitted model.

.. testcode::

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from ml_wrappers import wrap_model

    features = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    labels = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0).fit(features, labels)
    wrapped = wrap_model(model, features, model_task='classification')
    predictions = wrapped.predict(features)
    probabilities = wrapped.predict_proba(features)
    assert predictions.shape == (4,)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)

For regression, request ``model_task='regression'`` and call ``predict``.
The public function returns one model, not a ``(model, task)`` tuple.
See :ref:`model_wrapping` for task selection and output shapes.

Work with pandas data
---------------------

``DatasetWrapper`` exposes a DataFrame as a NumPy array and can reconstruct
its columns and dtypes after compatible modifications:

.. testcode::

    import pandas as pd
    from ml_wrappers import DatasetWrapper

    frame = pd.DataFrame({'age': [20, 30], 'income': [100, 200]})
    dataset = DatasetWrapper(frame)
    values = dataset.dataset.copy()
    values[:, 0] += 1
    restored = dataset.typed_wrapper_func(values)
    assert restored['age'].tolist() == [21, 31]
    assert restored.columns.tolist() == frame.columns.tolist()
    assert restored.dtypes.equals(frame.dtypes)
    assert dataset.typed_dataset.equals(frame)

``typed_dataset`` is a property, not a conversion method. See
:ref:`dataset_wrapping` for sampling, sparse inputs and conversion limitations.

Next steps
----------

Use :ref:`supported_models` to select an adapter, :ref:`API Reference` for
signatures, and :ref:`support` to report problems. The :ref:`privacy_policy`
page explains remote inference; the :ref:`license_information` page links
the MIT license.
