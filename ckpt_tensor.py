import tensorflow as tf
import argparse
import numpy as np
import os
import re

'''
Example: 
input_ckpt : model.ckpt-300000 
input_meta : model.ckpt-300000.meta
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input_ckpt', type=str, required=True,
                    default=None, help='CKPT path.')
parser.add_argument('--input_meta', type=str, 
                    default=None, help='meta path.')
parser.add_argument('--tensor_name', type=str, 
                    default=None, help='Tensor name, ex fc1/w.')
parser.add_argument('-s', '--show_nodes', type=bool, 
                    default=False, help='Show all op nodes')

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

def ensure_graph_is_valid(graph_def):
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map:
            node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)
    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            if input_node_name not in node_map:
                raise ValueError("Input for ", node.name, " not found: ",
                                 input_name)
    print(node_map)

def node_name_from_input(node_name):
   if node_name.startswith("^"):
       node_name = node_name[1:]
   m = re.search(r"(.*):\d+$", node_name)
   if m:
       node_name = m.group(1)
   return node_name

def main():

    # prepare the test data
    with tf.Session() as sess:

        # load the meta graph and weights
        saver = tf.train.import_meta_graph(args.input_meta)
        saver.restore(sess, args.input_ckpt)

        # get weights
        graph = tf.get_default_graph()
        if args.tensor_name:
            tensor = graph.get_tensor_by_name(args.tensor_name+":0")

            print(args.tensor_name)
            print("-"*100)
            print(sess.run(tensor))
            print("-"*100)
            pass

        if args.show_nodes:
            # nodes = [n for n in sess.graph_def.node if 'save' not in n.name]
            nodes = [n for n in sess.graph_def.node if 'masked_weight' in n.name and n.op != 'Const']
            for node in nodes:
                print(node)
            pass

        # ensure_graph_is_valid(graph.as_graph_def())

if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

