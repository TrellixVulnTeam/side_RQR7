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
    'data_type',
    'image',
    'Input Data Type: video, image'
)
flags.DEFINE_string(
    'data_path',
    'None',
    'Input Data PATH'
)
flags.DEFINE_string(
    'output_path',
    'None',
    'Output Data PATH'
)
FLAGS = flags.FLAGS


def Inference_pb(input_path, input_type, output_path, model_path):

    #= Parameter setting =#
    ti1 = tt.time()
    tt1 = 0
    c1 = 0
    c2 = 0
    frame_rate = 0
    # image_width = 1080
    # image_height = 1440
    # image_width = 314
    # image_height = 353
    # image_crop_w = 256
    # image_crop_h = 320
    need_rotation = False

    if output_path == 'None':
        output_path = '/'.join(input_path.split('/')[:-1])
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = output_path + '/{}'.format(input_path.split('/')[-1])
    print(output_path)
    
    #= Generating corresponding output path =#
    if input_type == 'video':
        cap = cv2.VideoCapture(input_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
        # 取得影像的尺寸大小
        image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        output_file = output_path.replace('.mp4', '_pb.mp4')

        # 使用 XVID 編碼，FPS 值為 20.0，解析度為 image_width x image_height
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(output_file, fourcc, 20.0, (image_width, image_height))

    elif input_type == 'image':
        pic = cv2.imread(input_path)
        image_height, image_width, channels = pic.shape
        # pic = cv2.resize(pic, (image_width, image_height))
        output_file = output_path.replace('.jpg', '_pb.jpg')

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
    h = int(i_tensor.shape[1]) # (N,H,W,C) [1] is height
    w = int(i_tensor.shape[2]) # (N,H,W,C) [2] is width

    print('\n Image (h, w): ({}, {})'.format(image_height, image_width))

    o_tensor = detection_graph.get_tensor_by_name('ONet/outputs:0')

    if input_type == 'video':
        while cap.isOpened():
            # Getting frame
            ret, frame_live = cap.read()
            # first_img = np.zeros((H, W, 3))
            if ret:
                frame_rgb = cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB)
                frame_np_expanded = np.expand_dims(frame_rgb, axis=0)   
                #=== pb operation ===
                t1 = tt.time()
                o_value = sess.run(o_tensor, feed_dict={i_tensor: frame_np_expanded}) # execution

                it1 = tt.time()-t1
                tt1 += it1
                c1 += 1
                if (tt.time()-ti1) >= 1:
                    ti1 = tt.time()
                    frame_rate = 1 / (tt1 / c1)
                    c1 = 0
                    tt1 = 0

                # image_live_np = frame_live.copy()
                # cv2.putText(image_live_np, '{:.1f}'.format(frame_rate), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                print(o_value)
                # === Video exporting
                video_out.write(image_live_np)
            else:
                break
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame = pic
        # output_frame = pic.copy()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_rgb = (np.float32(frame_rgb) - 127.5) / 128
        # frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
        # frame_np_expanded = np.expand_dims(frame_rgb, axis=0)

        output_frame, rotate_frame, frame_np_expanded = PreProcessing(frame, w, h,
            image_width, image_height, True)
        #=== pb operation ===
        t1 = tt.time()
        # o_value = sess.run(o_tensor, feed_dict={i_tensor: cropped}) # execution

        landmark_tensor = sess.run(o_tensor, feed_dict={i_tensor: frame_np_expanded}) # execution

        it1 = tt.time()-t1
        print("========== Inference time: {} ==========".format(it1))
        tt1 += it1
        c1 += 1
        if (tt.time()-ti1) >= 1:
            ti1 = tt.time()
            frame_rate = 1 / (tt1 / c1)
            c1 = 0
            tt1 = 0

        print("Landmark : \n{}\n".format(landmark_tensor))
        print("Landmark shape : {}\n".format(landmark_tensor.shape))
        # === Image exporting
        # oneD_array = landmark_tensor.reshape(-1)
        # np.savetxt(input_path.replace('.jpg', '_output.txt'), oneD_array, fmt="%.8f")
        output_frame = PostProcessing(pic, landmark_tensor, w, h,
            image_width, image_height)
        cv2.imwrite(output_file, output_frame)
        cv2.imwrite(output_file.replace('_pb.jpg', '_rotate.jpg'), rotate_frame)

        # oneD_array = frame_np_expanded.reshape(-1)
        # np.savetxt(input_path.replace('.jpg', '.txt'), oneD_array, fmt="%.8f")


    sess.close()

def PreProcessing(frame, crop_W, crop_H, image_W, image_H, rotation=False):
    frame = cv2.resize(frame, (crop_W, crop_H))
    ###########################
    if rotation:
        # frame_ = cv2.resize(frame, (crop_H, crop_W)) # width x height = image_H x image_W = x , y
        frame_ = cv2.flip(frame, 1)
        X = crop_H
        Y = crop_W
        M = cv2.getRotationMatrix2D((X/2 - 1, Y/2 - 1), 90, 1) # Counterclockwise 90
        M[0,2] += (Y - X) / 2 # axis-x shift
        M[1,2] += (X - Y) / 2 # axis-y shift
        frame_ = cv2.warpAffine(frame_, M, (crop_W, crop_H)) # width x height = image_W x image_H
        ## shift and crop
        # SH = np.float32([[1, 0, 0],[0, 1, -480]])
        # frame = cv2.warpAffine(frame, SH, (1080, 1440))
        r_frame = frame_
    else:
        frame_ = frame
        r_frame = frame
    ###########################
    # frame_reshape = frame.copy()
    frame_RGB = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    crop_image = frame_RGB
    # crop_image = cv2.resize(frame_RGB, (crop_W, crop_H), interpolation=cv2.INTER_AREA)
    crop_image = (crop_image - 127.5) * 0.0078125 # normalize to (-1, 1)
    crop_image = np.expand_dims(crop_image, 0)
    return frame, r_frame, crop_image   


def PostProcessing(image, landmarks, crop_W, crop_H, image_W, image_H):
    if image is None:
        return image
    landmarks = landmarks[0,:,0]
    # landmarks = (landmarks*48).astype(np.int) 
    landmarks[0:5] = (landmarks[0:5]*image_W)
    landmarks[5:10] = (landmarks[5:10]*image_H)
    landmarks = landmarks.astype(np.int)

    # scale_x = image_W / crop_W
    # scale_y = image_H / crop_H

    # image = cv2.resize(image, (crop_W, crop_H), interpolation=cv2.INTER_AREA)
    # image = cv2.flip(image, 1)
    image = cv2.UMat(image) # Using GPU acceleration if avaliable

    image = cv2.circle(image, (landmarks[0], landmarks[5]), 3, (0,255,255),-1)
    image = cv2.circle(image, (landmarks[1], landmarks[6]), 3, (0,255,255),-1)
    image = cv2.circle(image, (landmarks[2], landmarks[7]), 3, (0,255,255),-1)
    image = cv2.circle(image, (landmarks[3], landmarks[8]), 3, (0,255,255),-1)
    image = cv2.circle(image, (landmarks[4], landmarks[9]), 3, (0,255,255),-1)

    return image   


def main(_):

    # Recognize the correct model and data type
    # will need FLAGS.model, FLAGS.data_type, FLAGS.data_path
    model_name = FLAGS.model.split('/')[-1]
    model_type = model_name.split('.')[-1]
    print('Input data and model analyzing ...')
    try:
        if not model_type == 'pb':
            raise ValueError('!!! {} is not pb file. !!!'.format(FLAGS.model))
        Inference_pb(FLAGS.data_path, FLAGS.data_type, FLAGS.output_path, FLAGS.model)
        pass
    except ValueError as e:
        print("Error：",repr(e))
        pass

    print('Interprete finished !')


if __name__ == '__main__':
    tf.app.run()

