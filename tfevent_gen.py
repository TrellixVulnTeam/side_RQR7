import tensorflow as tf
import argparse

from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser()
parser.add_argument('--input_pb', type=str, 
                    default=None, help='PB model path.')
parser.add_argument('--output_path', type=str, 
                    default='None', help='tfevent path.')
parser.add_argument('--input_ckpt', type=str,
                    default='None', help='CKPT path.')
parser.add_argument('--input_meta', type=str,
                    default=None, help='meta path.')

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
   # load graph from pb or ckpt
    if args.input_pb is not None:
        with gfile.FastGFile(args.input_pb, 'rb') as f, \
            tf.Session() as sess:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='') # 导入计算图
            tf.summary.FileWriter(args.output_path, graph=tf.get_default_graph())
            print('='*15,'-Done-','='*15)
            print("Model Imported. Visualize by running: "
                  "tensorboard --logdir={}".format(args.output_path))
            print('Paste http://localhost:6006/.')
    if args.input_meta is not None:
        with tf.Graph().as_default() as g, tf.Session():
            tf.train.import_meta_graph(args.input_meta)
            file_writer = tf.summary.FileWriter(logdir=args.output_path, graph=g)
            print('='*15,'-Done-','='*15)
            print("Model Imported. Visualize by running: "
                  "tensorboard --logdir={}".format(args.output_path))
            print('Paste http://localhost:6006/.')
    # After done, tensorboard --logdir=args.output_path
    # visit: http://localhost:6006/

if __name__ == '__main__':
   main()
   # tf.app.run()

