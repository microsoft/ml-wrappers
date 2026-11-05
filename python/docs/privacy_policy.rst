.. _privacy_policy:

Data Processing and Privacy
===========================

Where inference runs depends on the model you wrap. Local dataset operations
and local model inference operate in your environment, but ML Wrappers does
not make an underlying model, framework or service local or private.

``EndpointWrapperModel`` sends inputs to the configured endpoint.
``OpenaiWrapperModel`` sends prompts and inputs through the configured OpenAI
or Azure OpenAI client. Wrapping can also invoke inference while evaluating
a model's output contract. Model/framework loading may download assets.

Review the underlying framework's behavior and the remote service's privacy,
retention, authentication and logging policies before passing data.
Use approved endpoints and datasets, and do not include credentials or
sensitive data in documentation examples or public issue reports.

For project support, see :ref:`support`. Report security vulnerabilities
through the private reporting process in
`SECURITY.md <https://github.com/microsoft/ml-wrappers/blob/main/SECURITY.md>`_,
not through public GitHub issues.
