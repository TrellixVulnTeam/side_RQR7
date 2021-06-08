"""
=== Rebuild model and export pb ===
inference model export
 python ckpt_export_pb.py \
--model '/workspace/tensorflow_lite/tf115/models/model.ckpt-975000' \
--output_graph 'PE_MOBILENET_V2_0.5_MSE_OHEM_F4_320_256_v2.pb' \
--model_type=SEMobilePose \
--backbone=mobilenet_v2 \
--input_node=cropped_image \
--output_nodes 'output/BiasAdd' \
--quantization=True # if training is quantization-awaring
model : the path of ckpt folder, must include 4 files:
    1. checkpoint 
    2. ckpt_name.data-00000-of-00001
    3. ckpt_name.index
    4. ckpt_name.meta
export file order
.pb
"""


import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import h5py


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'model',
    '../models/PE_MOBILENET_V1_0.5_MSE_COCO_320_256_v2/model.ckpt-430000',
    'CKPT PATH'
)
flags.DEFINE_string(
    'output_graph',
    'PE_MOBILENET_V1_0.5_MSE_COCO_320_256_v2.pb',
    'PB PATH'
)
flags.DEFINE_string(
    'model_type',
    'MobilePose',
    'Model architecture in [MobilePose, FPMobilePose, SEMobilePose]'
)
flags.DEFINE_string(
    'backbone',
    'mobilenet_v1',
    'Model backbone in [mobilenet_v1, mobilenet_v2, mobilenet_v3]'
)
flags.DEFINE_string(
    'input_node',
    'cropped_image',
    'Node name of input'
)
flags.DEFINE_string(
    'output_nodes',
    'output/BiasAdd',
    'Nodes of output, seperated by comma'
)
flags.DEFINE_boolean(
    'quantization',
    False,
    'Quantization Awaring'
)

FLAGS = flags.FLAGS


def save_and_frezze_model(sess,
                          checkpoint_path,
                          input_nodes,
                          output_nodes,
                          pb_path):
    print('======================================')
    # print('Saving .ckpt files...')
    saver = tf.train.Saver()
    # save light checkpoint file
    saver.save(sess, checkpoint_path)
    # print('***.ckpt files have been saving to', checkpoint_path)

    print('======================================')
    # print('Saving .pbtxt file...')
    graph_name = os.path.basename(pb_path).replace('.pb', '.pbtxt')
    graph_folder = os.path.dirname(pb_path)
    tf.train.write_graph(sess.graph.as_graph_def(),
                         graph_folder,
                         graph_name,
                         True)
    # print('***.pbtxt file has been saving to', graph_folder)

    ############# pb file saving #############
    print('======================================')
    print('Saving .pb file...')
    input_graph_path = os.path.join(graph_folder, graph_name)
    input_saver_def_path = ''
    input_binary = False
    input_checkpoint_path = checkpoint_path
    output_node_names = output_nodes
    restore_op_name = 'unused'
    filename_tensor_name = 'unused'
    output_graph_path = pb_path # pb file path
    clear_devices = True
    initializer_nodes = ''

    freeze_graph(input_graph_path, input_saver_def_path,
                 input_binary, input_checkpoint_path,
                 output_node_names, restore_op_name,
                 filename_tensor_name, output_graph_path,
                 clear_devices, initializer_nodes)
    print('***.pb file has been saving to', output_graph_path)
    os.remove(checkpoint_path+".data-00000-of-00001")
    os.remove(checkpoint_path+".index")
    os.remove(checkpoint_path+".meta")
    os.remove(input_graph_path)
    os.remove(graph_folder+"/checkpoint")

    print('======================================')
    ############# optimized pb file saving #############
    print('Saving optimized .pb file...')
    opt_graph_path = output_graph_path.replace('.pb', '_opt.pb') # optimized pb file path
    optimize_inference_model(output_graph_path,
                          opt_graph_path,
                          input_nodes,
                          output_nodes)



'''
optimize inference model
Remove the indentity ops that are in the original model.
'''
def optimize_inference_model(frozen_graph_path,
                             optimized_graph_path,
                             input_node_names,
                             output_node_names):
    print('Reading frozen graph...')
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_path, 'rb') as f:
        data2read = f.read()
        input_graph_def.ParseFromString(data2read)

    print('Optimizing frozen graph...')
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_node_names.split(','),  # an array of the input node(s)
        output_node_names.split(','),  # an array of the output nodes
        tf.float32.as_datatype_enum
    )

    print('Saving the optimized graph .pb file...')
    with tf.gfile.FastGFile(optimized_graph_path, 'w') as f:
        f.write(output_graph_def.SerializeToString())


def main(_):
    print('Rebuild graph...')
    if FLAGS.model_type == 'MobilePose':
        from mobilepose import MobilePose
        model_arch = MobilePose
    elif FLAGS.model_type == 'FPMobilePose':
        from fp_mobilepose import FPMobilePose
        model_arch = FPMobilePose
    elif FLAGS.model_type == 'SEMobilePose':
        from se_mobilepose import SEMobilePose
        model_arch = SEMobilePose
    else:
        print('{} not supported.'.format(FLAGS.model_type))
        return 0

    model = model_arch(backbone=FLAGS.backbone,
                       is_training=False,
                       depth_multiplier=0.5,
                       number_keypoints=17)            

    inputs = tf.placeholder(tf.float32,
                            shape=(None, 320, 256, 3),
                            name=FLAGS.input_node)
    end_points = model.build(inputs)

    if FLAGS.quantization:
        # Quantization-awaring eval inference
        g = tf.get_default_graph()
        # 将之前训练时构造的伪量化的operation和activation实际量化，用于后续的推断
        tf.contrib.quantize.create_eval_graph(g)

    sess = tf.Session()
    saver = tf.train.Saver()
    # restore checkpoint
    saver.restore(sess, FLAGS.model)
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
    variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    import numpy as np
    for i in variables:
        w = i
        # temp = (np.random.random(sess.run(i).shape) + 0.1) * 0.01
        temp = (np.ones(sess.run(i).shape)) * 0.001
        w.load(temp, sess)
        sess.run(w)

    output_path = '/'.join(FLAGS.model.split('/')[:-1]) + \
        '/output_pb/'
    output_files = output_path + 'model.ckpt'
    print('======= Output path =======: {}'.format(output_path))

    try:
        if FLAGS.output_graph.split('.')[-1] != 'pb':
            raise ValueError('!!! Output graph is incorrect format. !!!')
        print('Model exporting...')

        # save_and_frezze_model(sess,
        #                       output_files,
        #                       FLAGS.input_node,
        #                       FLAGS.output_nodes,
        #                       output_path + FLAGS.output_graph)
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), [FLAGS.output_nodes])
        tf.io.write_graph(frozen_graph, output_path, FLAGS.output_graph, as_text=False)

        print('Exporting finished !')
        pass
    except ValueError as e:
        print("Error：", repr(e))
        print(FLAGS.output_graph.split('.')[-1])
        pass
if __name__ == '__main__':
    tf.app.run()
