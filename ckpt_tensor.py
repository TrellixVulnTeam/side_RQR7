import tensorflow as tf
import argparse
import numpy as np
import os

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

def main():

    # prepare the test data
    with tf.Session() as sess:

        # load the meta graph and weights
        saver = tf.train.import_meta_graph(args.input_meta)
        saver.restore(sess, args.input_ckpt)

        # get weights
        graph = tf.get_default_graph()
        tensor = graph.get_tensor_by_name(args.tensor_name+":0")

        print(args.tensor_name)
        print("-"*100)
        print(sess.run(tensor))
        print("-"*100)

if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

