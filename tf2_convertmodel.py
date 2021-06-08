import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_pb', type=str, 
                    default='None', help='PB path.')
# parser.add_argument('--output_pb', type=str, 
#                     default='None', help='Output')


def frozen_keras_graph(func_model):
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func_model)

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != tf.resource
    ]
    output_tensors = frozen_func.outputs
    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold", "function"]),
        graph=frozen_func.graph)

    return graph_def


def convert_keras_model_to_pb():

    keras_model = train_model()
    func_model = tf.function(keras_model).get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    graph_def = frozen_keras_graph(func_model)
    tf.io.write_graph(graph_def, '/tmp/tf_model3', 'frozen_graph.pb')

def convert_saved_model_to_pb():
    model_dir = '/tmp/saved_model'
    model = tf.saved_model.load(model_dir)
    func_model = model.signatures["serving_default"]
    graph_def = frozen_keras_graph(func_model)
    tf.io.write_graph(graph_def, '/tmp/tf_model3', 'frozen_graph.pb')

