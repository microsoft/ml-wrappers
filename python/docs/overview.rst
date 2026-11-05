.. _overview:

Overview
========

ML Wrappers adapts supported datasets and models for responsible AI tools
that expect consistent prediction APIs. It does not train models or make
every framework interchangeable.

``DatasetWrapper`` exposes data and retains metadata for supported type
conversions. ``wrap_model`` selects task-specific adapters, including
tabular classification/regression, text and vision paths. Explicit wrappers
also support stored predictions and remote inference.

Start with :ref:`getting_started`, select a scenario in
:ref:`supported_models`, and check the canonical
:ref:`supported_frameworks` matrix before installing optional dependencies.
ONNX support is not implemented.

The ``latest`` site describes ``main``. See :ref:`versioning` for installed
releases, versioned documentation and the known older-tag build limitation.
