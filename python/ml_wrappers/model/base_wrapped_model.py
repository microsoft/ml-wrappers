# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a base class for wrapping models."""

from ml_wrappers.model.evaluator import _eval_model


class BaseWrappedModel(object):
    """A base class for WrappedClassificationModel and WrappedRegressionModel."""

    def __init__(self, model, eval_function, examples, model_task):
        """Initialize the WrappedClassificationModel with the model and evaluation function."""
        self._eval_function = eval_function
        self._model = model
        self._examples = examples
        self._model_task = model_task

    def __getstate__(self):
        """Influence how BaseWrappedModel is pickled.

        Removes _eval_function which may not be serializable.

        :return state: The state to be pickled, with _eval_function removed.
        :rtype: dict
        """
        odict = self.__dict__.copy()
        if self._examples is not None:
            del odict['_eval_function']
        return odict

    def __setstate__(self, state):
        """Influence how BaseWrappedModel is unpickled.

        Re-adds _eval_function which may not be serializable.

        :param dict: A dictionary of deserialized state.
        :type dict: dict
        """
        self.__dict__.update(state)
        if self._examples is not None:
            eval_function, _ = _eval_model(self._model, self._examples, self._model_task)
            self._eval_function = eval_function
