'''
Using this code for export the image or video that
are interpred tensorflow_lite.
cmd:
python frame_export.py \
--model 'model path' \
--data_type 'video, image' \
--data_path 'input data path' \
--output_path 'output data path (default is the same as input data)'

model path: '/workspace/tensorflow_lite/tf115/models/PE_SEN_ESPCN_MOBILENET_V2_0.5_MSE_OHEM_F4_320_256_v2_opt_quan8.tflite'

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
flags.DEFINE_string(
    'output_node',
    'None',
    'Output Node'
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
    image_crop_h = int(i_tensor.shape[1]) # (N,H,W,C) [1] is height
    image_crop_w = int(i_tensor.shape[2]) # (N,H,W,C) [2] is width
    print('\n crop',image_crop_w, image_crop_h)

    # o_tensor = detection_graph.get_tensor_by_name(FLAGS.output_node+':0')
    yaw_tensor = detection_graph.get_tensor_by_name('MobilenetV1_1/yaw/Softmax:0')
    pitch_tensor = detection_graph.get_tensor_by_name('MobilenetV1_1/pitch/Softmax:0')
    roll_tensor = detection_graph.get_tensor_by_name('MobilenetV1_1/roll/Softmax:0')

    if input_type == 'video':
        while cap.isOpened():
            # Getting frame
            ret, frame_live = cap.read()
            # first_img = np.zeros((H, W, 3))
            if ret:
                image_live_np, cropped = PreProcessing(frame_live, image_crop_w, image_crop_h, 
                                         image_width, image_height, need_rotation)                
                #=== pb operation ===
                t1 = tt.time()
                o_value = sess.run(o_tensor, feed_dict={i_tensor: cropped}) # execution

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
                # === Reverting frame
                PostProcessing(image_live_np, o_value, image_crop_w, image_crop_h)
                # === Video exporting
                video_out.write(image_live_np)
            else:
                break
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame_live = pic
        image_live_np, cropped = PreProcessing(frame_live, image_crop_w, image_crop_h, 
                                 image_width, image_height, need_rotation)                
        #=== pb operation ===
        t1 = tt.time()
        # o_value = sess.run(o_tensor, feed_dict={i_tensor: cropped}) # execution

        roll, pitch, yaw = sess.run([roll_tensor, pitch_tensor, yaw_tensor], feed_dict={i_tensor: cropped}) # execution

        it1 = tt.time()-t1
        print("========== Inference time: {} ==========".format(it1))
        tt1 += it1
        c1 += 1
        if (tt.time()-ti1) >= 1:
            ti1 = tt.time()
            frame_rate = 1 / (tt1 / c1)
            c1 = 0
            tt1 = 0

        # image_live_np = frame_live.copy()
        # cv2.putText(image_live_np, '{:.1f}'.format(frame_rate), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
        yaw_degree = 0
        pitch_degree = 0
        roll_degree = 0
        for i in range(0,66):
            yaw_degree += yaw[0][i] * i;
            pitch_degree += pitch[0][i] * i;
            roll_degree += roll[0][i] * i;
        yaw_degree = yaw_degree * 3 - 99;
        pitch_degree = pitch_degree * 3 - 99;
        roll_degree = roll_degree * 3 - 99;
        print("\n{} Degree : {}\n".format('roll', roll_degree))
        print("\n{} Degree : {}\n".format('pitch', pitch_degree))
        print("\n{} Degree : {}\n".format('yaw', yaw_degree))
        # === Reverting frame
        # PostProcessing(image_live_np, o_value, image_crop_w, image_crop_h)
        # === Image exporting
        # output_frame = (image_processed+1) * 255 /2
        # cv2.imwrite(output_file, image_live_np)


    sess.close()



# Reverting the position of mark and label the positions on frame 
def PostProcessing(frame, humanmarks, crop_W, crop_H):
    left_color = (255, 219, 160)
    right_color = (255, 255, 255)
    bingo_color = (191, 145, 73)
    thickness = 2
    antia = cv2.LINE_AA

    hm = humanmarks[0, :, :, :17].transpose(2, 0, 1)

    keypoints = []
    for i, part in enumerate(hm):
        ind = unravel_index(part.argmax(), part.shape) # 找到標注的點（有最大值）
        y, x = ind
        x = int(x * 4 * frame.shape[1] / crop_W)
        y = int(y * 4 * frame.shape[0] / crop_H)
        keypoints.append((x, y))

    center = (int((keypoints[5][0] + keypoints[6][0])/ 2), int((keypoints[5][1] + keypoints[6][1])/ 2))
    hip_center = (int((keypoints[11][0] + keypoints[12][0])/ 2), int((keypoints[11][1] + keypoints[12][1])/ 2))

    cv2.line(frame, (keypoints[5]), (keypoints[6]), left_color, thickness, antia)
    cv2.line(frame, (keypoints[5]), (keypoints[7]), right_color, thickness, antia)
    cv2.line(frame, (keypoints[7]), (keypoints[9]), right_color, thickness, antia)
    cv2.line(frame, (keypoints[11]), (keypoints[12]), left_color, thickness, antia)
    cv2.line(frame, (keypoints[11]), (keypoints[13]), right_color, thickness, antia)
    cv2.line(frame, (keypoints[13]), (keypoints[15]), right_color, thickness, antia)
    cv2.line(frame, (keypoints[6]), (keypoints[8]), left_color, thickness, antia)
    cv2.line(frame, (keypoints[8]), (keypoints[10]), left_color, thickness, antia)
    cv2.line(frame, (keypoints[12]), (keypoints[14]), left_color, thickness, antia)
    cv2.line(frame, (keypoints[14]), (keypoints[16]), left_color, thickness, antia)
    cv2.line(frame, center, hip_center, left_color, thickness, antia)

# Return background frame(frame) and cropped frame for inference(crop_image).
def PreProcessing(frame, crop_W, crop_H, image_W, image_H, rotation=False):
    frame = cv2.resize(frame, (image_W, image_H))
    ###########################
    if rotation:
        frame = cv2.resize(frame, (image_H, image_W)) # width x height = image_H x image_W = x , y
        X = image_H
        Y = image_W
        M = cv2.getRotationMatrix2D((X/2 - 1, Y/2 - 1), 90, 1) # (x, y) of frame
        M[0,2] += (Y - X) / 2 # axis-x shift
        M[1,2] += (X - Y) / 2 # axis-y shift
        frame = cv2.warpAffine(frame, M, (image_W, image_H)) # width x height = image_W x image_H
        ## shift and crop
        # SH = np.float32([[1, 0, 0],[0, 1, -480]])
        # frame = cv2.warpAffine(frame, SH, (1080, 1440))
    ###########################
    # frame_reshape = frame.copy()
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    crop_image = cv2.resize(frame_RGB, (crop_W, crop_H), interpolation=cv2.INTER_AREA)
    crop_image = (crop_image - 127.5) * 0.0078125 # normalize to (-1, 1)
    crop_image = np.expand_dims(crop_image, 0)
    return frame, crop_image      


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

