import tensorflow as tf
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_pb', type=str, 
                    default='None', help='PB path.')
# parser.add_argument('--output_pb', type=str, 
#                     default='None', help='Output')

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
    output_path = '/'.join(args.input_pb.split('/')[:-1]) + '/'
    output_pb = args.input_pb.split('/')[-1].replace('.pb', '_init.pb')
    with pb_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.input_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with pb_graph.as_default():
    with tf.Session(graph=pb_graph, config=config) as sess:

        tensor_conv = [i.name for i in pb_graph.get_operations() if 'Conv' in i.name]
        _tensor = pb_graph.get_tensor_by_name(tensor_conv[0]+':0')
        print("-"*30+"Raw weight"+"-"*30)
        tensor = sess.run(_tensor)
        print(tensor[0])
    # ===== initial weight, will let the weight be initialized =====
        sess.run(tf.global_variables_initializer())

    # ===== make all variables random =====
        # variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        # import numpy as np
        # for i in variables:
        #     w = i
        #     # temp = (np.random.random(sess.run(i).shape) + 0.1) * 0.01
        #     temp = (np.ones(sess.run(i).shape)) * 0.001
        #     w.load(temp, sess)
        #     sess.run(w)
        tf.io.write_graph(pb_graph, output_path, output_pb, as_text=False)
        _init_tensor = pb_graph.get_tensor_by_name(tensor_conv[0]+':0')
        print("-"*30+"Init weight"+"-"*30)
        init_tensor = sess.run(_init_tensor)
        print(init_tensor[0])


if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

