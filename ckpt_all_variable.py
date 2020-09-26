import tensorflow as tf
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_ckpt', type=str, 
                    default='None', help='CKPT path.')
parser.add_argument('--input_meta', type=str, 
                    default='None', help='meta path.')
parser.add_argument('--tensor_name', type=str, 
                    default='None', help='Tensor name, ex fc1/w.')

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

def main(_):
    # build graph
    graph = tf.Graph
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')
    net, end_points = vgg.vgg_16(inputs, num_classes=1000)
    saver = tf.train.Saver()

    saver = tf.train.import_meta_graph(args.input_meta)

    with tf.Session() as sess:
        saver.restore(sess, args.input_ckpt) 
        """
           查看恢复的模型参数
           tf.trainable_variables()查看的是所有可训练的变量；
           tf.global_variables()获得的与tf.trainable_variables()类似，只是多了一些非trainable的变量，比如定义时指定为trainable=False的变量；
           sess.graph.get_operations()则可以获得几乎所有的operations相关的tensor
           """
        tvs = [v for v in tf.trainable_variables()]
        print('获得所有可训练变量的权重:')
        for v in tvs:
            print(v.name)
            print(sess.run(v))
        
        gv = [v for v in tf.global_variables()]
        print('获得所有变量:')
        for v in gv:
            print(v.name, '\n')
        
        # sess.graph.get_operations()可以换为tf.get_default_graph().get_operations()
        ops = [o for o in sess.graph.get_operations()]
        print('获得所有operations相关的tensor:')
        for o in ops:
            print(o.name, '\n')

if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

