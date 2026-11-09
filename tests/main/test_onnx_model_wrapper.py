# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Exercise ONNX adaptation using locally generated, real runtime sessions."""

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
from ml_wrappers import wrap_model
from ml_wrappers.dataset import DatasetWrapper
from ml_wrappers.model import WrappedOnnxModel
from onnx import TensorProto, helper
from skl2onnx import to_onnx
from sklearn.linear_model import LogisticRegression


def _session(graph):
    opsets = [helper.make_opsetid('', 18)]
    if any(node.domain == 'ai.onnx.ml' for node in graph.node):
        opsets.append(helper.make_opsetid('ai.onnx.ml', 3))
    model = helper.make_model(graph, opset_imports=opsets, ir_version=8)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options,
                                providers=['CPUExecutionProvider'])


def _regressor(dtype=TensorProto.FLOAT, keepdims=1, inputs=1):
    graph_inputs = [
        helper.make_tensor_value_info('input' + str(i), dtype, [None, 2])
        for i in range(inputs)
    ]
    return _session(helper.make_graph(
        [helper.make_node('Cast', ['input0'], ['features'], to=TensorProto.FLOAT),
         helper.make_node('ReduceSum', ['features', 'axes'], ['output'], keepdims=keepdims)],
        'regressor', graph_inputs,
        [helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [None, 1] if keepdims else [None])],
        [helper.make_tensor('axes', TensorProto.INT64, [1], [1])]))


@pytest.mark.parametrize('keepdims', [0, 1])
@pytest.mark.parametrize('dtype', [TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT64])
@pytest.mark.parametrize('use_dataframe', [False, True])
def test_regression(keepdims, dtype, use_dataframe):
    session = _regressor(dtype, keepdims)
    data = np.array([[1., 2.], [3., 4.], [-1., -2.]])
    examples = pd.DataFrame(data, columns=['a', 'b']) if use_dataframe else data
    wrapped = wrap_model(session, DatasetWrapper(examples), model_task='regression')
    np.testing.assert_allclose(wrapped.predict(examples), [3., 7., -3.])
    np.testing.assert_allclose(wrapped.predict(data[:1]), [3.])
    np.testing.assert_allclose(wrapped.predict(data[0]), [3.])
    if not keepdims:
        inferred = wrap_model(session, examples)
        np.testing.assert_allclose(inferred.predict(examples), [3., 7., -3.])
    else:
        with pytest.raises(Exception, match='specify model_task'):
            wrap_model(session, examples)


@pytest.mark.parametrize('zipmap', [False, True])
@pytest.mark.parametrize('labels', [[-5, 20], ['cat', 'dog']])
@pytest.mark.parametrize('include_labels', [False, True])
def test_exported_classifier(zipmap, labels, include_labels):
    data = np.array([[-3., -2.], [-1., -1.], [1., 1.], [3., 2.]], dtype=np.float32)
    target = np.array([labels[0], labels[0], labels[1], labels[1]])
    model = LogisticRegression().fit(data, target)
    converted = to_onnx(model, data[:1], options={'zipmap': zipmap}, target_opset=18)
    if not include_labels:
        del converted.graph.output[0]
    session = ort.InferenceSession(converted.SerializeToString(),
                                   providers=['CPUExecutionProvider'])
    examples = pd.DataFrame(data, columns=['a', 'b'])
    for task in ['unknown', 'classification']:
        wrapped = wrap_model(session, examples, model_task=task)
        expected_labels = model.predict(data) if include_labels or zipmap else [0, 0, 1, 1]
        np.testing.assert_array_equal(wrapped.predict(examples), expected_labels)
        np.testing.assert_allclose(
            wrapped.predict_proba(examples), model.predict_proba(data), atol=1e-6)
        np.testing.assert_array_equal(wrapped.predict(data[0]), expected_labels[:1])
    if include_labels:
        with pytest.raises(ValueError, match='regression requires a single'):
            wrap_model(session, examples, model_task='regression')
    elif zipmap:
        with pytest.raises(ValueError, match='only supported for classification'):
            wrap_model(session, examples, model_task='regression')


def test_tensor_only_classifier():
    session = _session(helper.make_graph(
        [helper.make_node('Softmax', ['input'], ['probabilities'], axis=1)],
        'classifier',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, 3])]))
    data = np.array([[1., 2., 3.], [3., 2., 1.]])
    wrapped = wrap_model(session, data)
    expected = session.run(None, {'input': data.astype(np.float32)})[0]
    np.testing.assert_allclose(wrapped.predict_proba(data), expected)
    np.testing.assert_array_equal(wrapped.predict(data), [2, 0])
    with pytest.raises(Exception, match='Regression function cannot output'):
        wrap_model(session, data, model_task='regression')


@pytest.mark.parametrize('include_labels', [False, True])
def test_reject_single_class_zipmap(include_labels):
    map_type = helper.make_map_type_proto(
        TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
    outputs = [helper.make_value_info(
        'probabilities', helper.make_sequence_type_proto(map_type))]
    nodes = [helper.make_node('ZipMap', ['input'], ['probabilities'],
                              domain='ai.onnx.ml', classlabels_int64s=[7])]
    initializers = []
    if include_labels:
        nodes.extend([
            helper.make_node('ArgMax', ['input'], ['index'], axis=1, keepdims=0),
            helper.make_node('Add', ['index', 'label'], ['labels'])
        ])
        outputs.append(helper.make_tensor_value_info('labels', TensorProto.INT64, [None]))
        initializers.append(helper.make_tensor('label', TensorProto.INT64, [], [7]))
    session = _session(helper.make_graph(
        nodes, 'single_class_zipmap',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])],
        outputs, initializers))
    data = np.array([[0.2], [0.8], [1.0]], dtype=np.float32)
    for task in ['classification', 'unknown']:
        with pytest.raises(ValueError, match='probability column per class'):
            wrap_model(session, data, model_task=task)


def test_unsupported_schemas():
    with pytest.raises(ValueError, match='one rank-two tensor input'):
        wrap_model(_regressor(inputs=2), np.ones((2, 2)), model_task='regression')
    with pytest.raises(ValueError, match='probability column per class'):
        wrap_model(_regressor(), np.ones((2, 2)), model_task='classification')
    with pytest.raises(ValueError, match='dense matrix'):
        WrappedOnnxModel(_regressor()).predict(np.ones((1, 1, 2)))
    session = _session(helper.make_graph(
        [helper.make_node('Identity', ['input'], ['labels'])], 'labels_only',
        [helper.make_tensor_value_info('input', TensorProto.INT64, [None, 2])],
        [helper.make_tensor_value_info('labels', TensorProto.INT64, [None, 2])]))
    with pytest.raises(ValueError, match='floating-point'):
        wrap_model(session, np.ones((2, 2)), model_task='classification')
