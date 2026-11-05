# Wrapper specifications

The canonical [model wrapper contract](https://ml-wrappers.readthedocs.io/en/latest/model_wrapper_specifications.html)
describes the implemented API, task values, input requirements and output shapes.
See the [source](../python/docs/model_wrapper_specifications.rst) when reviewing
changes that have not yet been published to Read the Docs.

Use `wrap_model(model, examples, model_task='classification')` or
`model_task='regression'`; the default is `'unknown'`. The public function
returns one wrapped model, not an internal `(model, task)` tuple.
Text and vision have separate task-specific adapters and contracts.

See the [quickstart](https://ml-wrappers.readthedocs.io/en/latest/getting_started.html)
for executable examples and the [support matrix](https://ml-wrappers.readthedocs.io/en/latest/supported_frameworks.html)
for implemented scenarios. ONNX is not currently supported.
