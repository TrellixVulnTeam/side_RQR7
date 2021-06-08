'''
Using this code for Head Pose pb inference.
cmd:
python /workspace/side/pb_inference_test.py --model /workspace/tpu_rel_v1.2/HP/HP.pb --data_path /workspace/tpu_rel_v1.2/testimgs/hp_03.jpg 
'''
import tensorflow as tf
import numpy as np
import sys
from numpy import unravel_index
import cv2
from time import time
from math import acos
import math
import time as tt
import threading
import timeit
import os

flags = tf.app.flags
flags.DEFINE_string(
    'model',
    'None',
    'Model PATH'
)
flags.DEFINE_string(
    'data_path',
    'None',
    'Input Data PATH'
)

FLAGS = flags.FLAGS


def Inference_pb(input_path, model_path=None):

    #= Parameter setting =#
    ti1 = tt.time()
    tt1 = 0
    c1 = 0
    c2 = 0
    frame_rate = 0
    need_rotation = False
    
    pic = cv2.imread(input_path)
    image_height, image_width, channels = pic.shape
    # pic = cv2.resize(pic, (image_width, image_height))

    #= Setting pb model =#
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            print('='*10, 'Check out the input placeholders:', '='*10)
            nodes = [n.name for n in od_graph_def.node if n.op in ('Placeholder')]
            for node in nodes:
                print(node)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    sess = tf.Session(graph=detection_graph, config=config)
    print('Input node: ', nodes[0], '\n')
    i_tensor = detection_graph.get_tensor_by_name(nodes[0]+':0')
    image_crop_h = int(i_tensor.shape[1]) # (N,H,W,C) [1] is height
    image_crop_w = int(i_tensor.shape[2]) # (N,H,W,C) [2] is width
    print('\n Image (h, w): ({}, {})'.format(image_height, image_width))

    features_tensor = detection_graph.get_tensor_by_name('mobilefacenet/embedding:0')

    frame = pic
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = (np.float32(frame_rgb) - 127.5) / 128
    frame_rgb = cv2.resize(frame_rgb, (image_crop_w, image_crop_h), interpolation=cv2.INTER_AREA)
    frame_np_expanded = np.expand_dims(frame_rgb, axis=0)
    #=== pb operation ===
    t1 = tt.time()
    # o_value = sess.run(o_tensor, feed_dict={i_tensor: cropped}) # execution

    features = sess.run([features_tensor],
        feed_dict={i_tensor: frame_np_expanded}) # execution

    it1 = tt.time()-t1
    print("========== Inference time: {} ==========".format(it1))
    tt1 += it1
    c1 += 1
    if (tt.time()-ti1) >= 1:
        ti1 = tt.time()
        frame_rate = 1 / (tt1 / c1)
        c1 = 0
        tt1 = 0

    features = features[0]
    np.set_printoptions(suppress=True)
    print(features)
    # === Image exporting
    # output_frame = frame[x1:x2, y1:y2]
    # cv2.imwrite(output_file, output_frame)

    # oneD_array = frame_np_expanded.reshape(-1)
    # np.savetxt(input_path.replace('.jpg', '.txt'), oneD_array, fmt="%.8f")
    # np.savetxt(input_path.replace('.jpg', '_uint.txt'), oneD_array, fmt="%.5d")
    # np.savetxt(input_path.replace('.jpg', '_output.txt'), features[0], fmt="%.8f")


    sess.close()

def main(_):

    # Recognize the correct model and data type
    # will need FLAGS.model, FLAGS.data_type, FLAGS.data_path
    model_name = FLAGS.model.split('/')[-1]
    model_type = model_name.split('.')[-1]
    print('Input data and model analyzing ...')
    try:
        if not model_type == 'pb':
            raise ValueError('!!! {} is not pb file. !!!'.format(FLAGS.model))
        Inference_pb(FLAGS.data_path, FLAGS.model)
        pass
    except ValueError as e:
        print("Error：",repr(e))
        pass

    print('Interprete finished !')


if __name__ == '__main__':
    tf.app.run()

