# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Adapt single-input tabular ONNX Runtime sessions."""

import numpy as np

from ..common.constants import ModelTask


def _is_onnx_model(model):
    """Recognize ONNX Runtime sessions without importing an optional runtime."""
    if not any(base.__module__.startswith('onnxruntime.')
               for base in type(model).__mro__):
        return False
    from onnxruntime import InferenceSession
    return isinstance(model, InferenceSession)


class WrappedOnnxModel:
    """Provide prediction methods for an ONNX Runtime InferenceSession.

    Supports one rank-two dense tensor input and either one floating-point
    prediction tensor or a label vector plus class probabilities. Probabilities
    may be a matrix or an ONNX-ML ZipMap sequence. Tensor-only classifiers use
    column indices as labels; exported label outputs and ZipMap keys retain
    their original labels. No softmax or calibration is applied.
    """

    def __init__(self, model, model_task=ModelTask.UNKNOWN):
        """Initialize from a session configured with the desired providers.

        :param model: An ONNX Runtime InferenceSession.
        :param model_task: Tabular classification, regression, or unknown.
        """
        inputs = model.get_inputs()
        if len(inputs) != 1 or inputs[0].shape is None or len(inputs[0].shape) != 2:
            raise ValueError("ONNX models require one rank-two tensor input.")
        dtypes = {
            'float': np.float32, 'double': np.float64, 'float16': np.float16,
            'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
            'int64': np.int64, 'uint8': np.uint8, 'uint16': np.uint16,
            'uint32': np.uint32, 'uint64': np.uint64, 'bool': np.bool_,
            'string': object
        }
        self._input = inputs[0]
        self._dtype = dtypes.get(self._input.type.removeprefix('tensor(').removesuffix(')'))
        if self._dtype is None:
            raise ValueError("Unsupported ONNX input type: " + self._input.type)
        outputs = model.get_outputs()
        float_types = ('tensor(float)', 'tensor(double)', 'tensor(float16)')
        probabilities = [
            output for output in outputs
            if output.type.startswith('seq(map(') or (
                output.type in float_types and output.shape is not None and len(output.shape) == 2)
        ]
        self._labels = None
        if len(outputs) == 1:
            self._output = outputs[0]
        elif len(outputs) == 2 and len(probabilities) == 1:
            self._output = probabilities[0]
            self._labels = next(output for output in outputs if output != self._output)
            shape = self._labels.shape
            if not self._labels.type.startswith('tensor(') or shape is None or len(shape) != 1:
                raise ValueError("ONNX classification labels must be a vector.")
            if model_task == ModelTask.REGRESSION:
                raise ValueError("ONNX regression requires a single prediction output.")
        else:
            raise ValueError("ONNX models require one prediction output or labels and probabilities.")
        self._is_map = self._output.type.startswith('seq(map(')
        shape = self._output.shape
        if not self._is_map and (
                self._output.type not in float_types or shape is None or len(shape) not in (1, 2)):
            raise ValueError("ONNX predictions must be a floating-point vector or matrix, or ZipMap.")
        if self._is_map and model_task == ModelTask.REGRESSION:
            raise ValueError("ONNX ZipMap output is only supported for classification.")
        self._model = model
        self._model_task = model_task

    def _run(self, dataset, output_name):
        values = np.asarray(dataset, dtype=self._dtype)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("ONNX prediction inputs must be a dense matrix or a single row.")
        result = self._model.run([output_name], {self._input.name: values})[0]
        if len(result) != len(values):
            raise ValueError("ONNX predictions must contain one result per input row.")
        return result

    def _probabilities(self, dataset):
        result = self._run(dataset, self._output.name)
        classes = None
        if self._is_map:
            if not result:
                raise ValueError("ONNX ZipMap output must contain at least one row.")
            classes = sorted(result[0])
            if not classes or any(set(row) != set(classes) for row in result):
                raise ValueError("ONNX ZipMap rows must have the same nonempty class keys.")
            result = np.asarray([[row[label] for label in classes] for row in result])
        else:
            result = np.asarray(result)
        is_classifier = self._model_task == ModelTask.CLASSIFICATION or self._labels is not None or self._is_map
        if is_classifier and (result.ndim != 2 or result.shape[1] < 2):
            raise ValueError("ONNX classification requires a probability column per class and at least two classes; "
                             "single-column outputs are not supported.")
        return result, classes

    def predict(self, dataset):
        """Return regression values or classification labels for dense inputs."""
        if self._labels is not None:
            return np.asarray(self._run(dataset, self._labels.name))
        values, classes = self._probabilities(dataset)
        is_single_output = values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1)
        if self._model_task == ModelTask.REGRESSION or (self._model_task == ModelTask.UNKNOWN and is_single_output):
            return values
        indices = np.argmax(values, axis=1)
        return indices if classes is None else np.asarray(classes)[indices]

    def predict_proba(self, dataset):
        """Return the raw prediction tensor or dense ZipMap probabilities."""
        return self._probabilities(dataset)[0]
