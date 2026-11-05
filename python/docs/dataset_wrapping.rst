.. _dataset_wrapping:

Dataset Wrapping
================

``DatasetWrapper`` retains input metadata while exposing data for common
operations. Pandas DataFrames and Series expose their NumPy values through
``dataset``; NumPy arrays and SciPy sparse inputs are retained.

Restore a modified array
------------------------

Use ``typed_wrapper_func(array)`` for a separately modified array.
``typed_dataset`` instead restores the wrapper's current data and is a
property, not a callable.

.. testcode::

    import pandas as pd
    from ml_wrappers import DatasetWrapper

    original = pd.DataFrame({'age': [20, 30], 'income': [100, 200]})
    wrapper = DatasetWrapper(original)
    values = wrapper.dataset.copy()
    values[:, 0] += 1
    restored = wrapper.typed_wrapper_func(values)
    assert isinstance(restored, pd.DataFrame)
    assert restored['age'].tolist() == [21, 31]
    assert restored.dtypes.equals(original.dtypes)
    assert wrapper.typed_dataset.equals(original)

DataFrame reconstruction uses the original column names and dtypes. Keep a
compatible shape and values: casting back to the original dtype can reject
or truncate incompatible changes. Arbitrary transformations are not
automatically reversible. A Series is reconstructed from values; do not
assume all original Series metadata is retained. Non-default DataFrame
indexes require the wrapper's index-handling methods; an arbitrary array
does not carry an index.

Sampling and summaries
----------------------

The public sampling method is ``sample()``, not ``sample_examples()``.
It returns sampled data and, when sampling takes place, replaces the
wrapper's stored dataset. Keep the original data separately if needed.

.. testcode::

    small = DatasetWrapper(original)
    sampled = small.sample()
    assert sampled.shape == (2, 2)

Small datasets are returned unchanged. Larger datasets can use HDBSCAN
(an optional dependency) or the k-means fallback. ``compute_summary`` also
replaces the stored data with a summary; consult :ref:`API Reference` before
using transformations that change its shape.

Framework inputs
-----------------

The wrapper also recognizes SHAP DenseData, PyTorch tensors and certain
TensorFlow BatchDataset objects. These require their frameworks and are
not all eagerly converted into NumPy/SciPy by the constructor. TensorFlow
batch conversion has a framework-specific reconstruction path. Do not
assume every accepted input has the same conversion or metadata guarantees.

Avoid private lifecycle methods such as ``_clear`` in application code.
