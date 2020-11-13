import tensorflow as tf
import argparse
import numpy as np
import os

from tensorflow.contrib import graph_editor as ge

'''
Example: 
input_ckpt : model.ckpt-300000 
input_meta : model.ckpt-300000.meta
'''
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input_ckpt', type=str, required=True,
                    default=None, help='CKPT path.')
parser.add_argument('--input_meta', type=str, 
                    default=None, help='meta path.')
parser.add_argument('--tensor_name', type=str, 
                    default=None, help='Tensor name, ex fc1/w.')

args = parser.parse_args()

'''
flags = tf.app.flags 
flags.DEFINE_string(
    'train_data_path',
    # '/datasets/data1/faces_ms1m_112x112/intermediate/shuffle_faces_ms1m_112x112.record', #gtx
    '/datasets/t1/data/faces_ms1m_112x112/intermediate/shuffle_faces_ms1m_112x112.record', #rtx
    'Input Training dataset path'
    )
FLAGS = flags.FLAGS
'''

if args.input_meta is None:
    args.input_meta = args.input_ckpt + '.meta'
    pass

def show_ops(graph):

    # ge_ops = print(ge.make_list_of_op(graph))
    # ge_tensors = ge.make_list_of_t(graph)
    # print(ge.get_generating_ops(ge_tensors))

    # get subgraph by scope
    my_sgv = ge.sgv_scope('MobileFaceNet/expanded_conv_5', graph=graph)
    print("-"*100)
    # print(ge.make_list_of_op(my_sgv.ops))
    _, outputs = ge.detach_outputs(my_sgv)
    _, inputs = ge.detach_inputs(my_sgv)
    print(outputs)
    print("-"*100)
    print(inputs)
    # ge_tensors = ge.make_list_of_t(my_sgv)
    # print(ge.get_generating_ops(ge_tensors))
    pass

def save_pb(sess, graph):
    graph_folder = './'
    graph_name = 'test.pb'
    # graph_, _ = ge.detach_inputs(graph)
    # graph_ = ge.make_placeholder(graph, scope='input', dtype=tf.float32, shape=(1, 112, 112, 3))
    # frozen_graph = tf.graph_util.convert_variables_to_constants(
    #     sess, graph_.as_graph_def(), ['Logits/LinearConv1x1/BiasAdd'])
    # tf.train.write_graph(frozen_graph,
    #                      graph_folder,
    #                      graph_name,
    #                      False)
    
    '''
    with graph.as_default():
        ge.make_placeholder_from_dtype_and_shape(scope='input', dtype=tf.float32, shape=(1, 112, 112, 3))
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['Logits/LinearConv1x1/BiasAdd'])
        tf.train.write_graph(frozen_graph,
                             graph_folder,
                             graph_name,
                             False)
    '''

    #=== Bypass subgraph ===
    my_sgv = ge.sgv_scope('MobileFaceNet/expanded_conv_5', graph=graph)
    my_sgv = ge.bypass(my_sgv)
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ['Logits/LinearConv1x1/BiasAdd'])
    tf.train.write_graph(frozen_graph,
                         graph_folder,
                         graph_name,
                         False)



def main():

    # prepare the test data
    with tf.Session() as sess:

        # load the meta graph and weights
        saver = tf.train.import_meta_graph(args.input_meta)
        saver.restore(sess, args.input_ckpt)

        # get graph
        graph = tf.get_default_graph()
        show_ops(graph)
        # save_pb(sess, graph)


if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

