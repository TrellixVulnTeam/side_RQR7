import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import h5py
import argparse
import numpy as np
import os

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
parser.add_argument('--output_node', type=str, required=True,
                    default=None, help='Output node name, ex fc1/w.')
parser.add_argument('--output_path', type=str, 
                    default=None, help='Output dir path.')

args = parser.parse_args()


flags = tf.app.flags
flags.DEFINE_boolean(
    'quantization',
    False,
    'Quantization Awaring'
)
FLAGS = flags.FLAGS


if args.input_meta is None:
    args.input_meta = args.input_ckpt + '.meta'
    pass
if args.output_path is None:    
    args.output_path = os.path.dirname(args.input_ckpt)
    pass
args.output_node = [args.output_node]


def main(_):
    graph_folder = args.output_path
    graph_name = 'output.pb'

    print('Rebuild graph...')

    sess = tf.Session()

    inputs = tf.placeholder(tf.float32,
                            shape=(None, 320, 256, 3),
                            name='input')

    # load the meta graph
    saver = tf.train.import_meta_graph(args.input_meta, input_map={'MobilenetV2/input': inputs})
    # saver = tf.train.import_meta_graph(args.input_meta)

    # get graph
    graph = tf.get_default_graph()


    if FLAGS.quantization:
        # 将之前训练时构造的伪量化的operation和activation实际量化，用于后续的推断
        tf.contrib.quantize.create_eval_graph(graph)

    # restore checkpoint
    saver.restore(sess, args.input_ckpt)
    '''
    # ===== initial weight, will let the weight be initialized =====
    sess.run(tf.global_variables_initializer()) 

    # ===== modify variables =====
    with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
        # w = sess.run(tf.get_variable("bias", shape=(17,), dtype=tf.float32, initializer = tf.random_normal_initializer()))               
        w = tf.get_variable("bias", shape=(17,), dtype=tf.float32)               
        w.load([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], sess)
        print(sess.run(w))
    '''
    # ===== make all variables random =====
    '''
    variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    import numpy as np
    for i in variables:
        w = i
        # temp = (np.random.random(sess.run(i).shape) + 0.1) * 0.01
        temp = (np.ones(sess.run(i).shape)) * 0.001
        w.load(temp, sess)
        sess.run(w)
    '''

    print('======= Output path =======: {}'.format(graph_folder + '/' + graph_name))

    try:
        print('Model exporting...')

        with graph.as_default():
            # ge.make_placeholder_from_dtype_and_shape(scope='input', dtype=tf.float32, shape=(1, 112, 112, 3))
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), args.output_node)
            tf.train.write_graph(frozen_graph,
                                 graph_folder,
                                 graph_name,
                                 False)
        print('Exporting finished !')
        pass
    except ValueError as e:
        print("Error：", repr(e))
        pass
if __name__ == '__main__':
    tf.app.run()
