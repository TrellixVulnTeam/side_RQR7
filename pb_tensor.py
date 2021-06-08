import tensorflow as tf
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_pb', type=str, 
                    default=None, help='PB path.')
parser.add_argument('--tensor_name', type=str, 
                    default=None, help='Tensor name, ex weight')
parser.add_argument('--scope_name', type=str, 
                    default=None, help='Scope name')
parser.add_argument(
    "--all_tensor_names",
    nargs="?",
    const=True,
    type=bool,
    default=False,
    help="If True, print the names of all the tensors.")

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

def main():

    pb_graph = tf.Graph()
    with pb_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.input_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with pb_graph.as_default():
        with tf.Session(graph=pb_graph, config=config) as sess:

            print('='*10, 'Check out the input placeholders:', '='*10)
            nodes = [n.name + ' => ' +  n.op for n in sess.graph_def.node if n.op in ('Placeholder')]
            for node in nodes:
                print(node)
        # ===Get by Name===
            # _tensor = pb_graph.get_tensor_by_name(args.tensor_name+':0')
            # print(sess.run(_tensor))
        # === Get and print all tensor name ===
            if args.all_tensor_names:
                for i in pb_graph.get_operations():
                    print('tensor name: {}\r'.format(i.name))
        # ===Get collection by op that include the tensor_name===
            if args.tensor_name is not None:
                tensor_collections = [i.name for i in pb_graph.get_operations() if args.tensor_name in i.name]
                for x in tensor_collections:
                    _tensor = pb_graph.get_tensor_by_name(x+':0')
                    print("-"*100)
                    tensor = sess.run(_tensor)
                    print(tensor.shape)
                # tensor detail
                    # T_shape = _tensor.shape
                    # T_dtype = _tensor.dtype
                    # T_name = _tensor.name
                    # T_op = _tensor.op
        # === Get scope name ===
            if args.scope_name is not None:
                tensor_collections = [i.name for i in pb_graph.get_operations() if args.scope_name in i.name]
                for x in tensor_collections:
                    print('tensor name: {}\r'.format(x))
                
if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

