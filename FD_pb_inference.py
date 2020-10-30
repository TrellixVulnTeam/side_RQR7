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
    print('\n Image (h, w): ({}, {})'.format(image_height, image_width))

    FD_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    FD_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # FD_num = detection_graph.get_tensor_by_name('num_detections:0')
    # FD_classes = detection_graph.get_tensor_by_name('detection_classes:0')

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_np_expanded = np.expand_dims(frame_rgb, axis=0)
        #=== pb operation ===
        t1 = tt.time()
        # o_value = sess.run(o_tensor, feed_dict={i_tensor: cropped}) # execution

        boxes,scores = sess.run([FD_boxes, FD_scores],
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

        max_i = np.argmax(scores)
        box_coord = boxes[0][max_i] * np.array([image_width,image_height,image_width,image_height])
        x1, y1, x2, y2 = box_coord.astype('int')
        # cv2.rectangle(image_live_np,(y1,x1),(y2,x2),(0,255,255),4)
        print("Boxes : {}\n".format(boxes[0][max_i]))
        print("Boxes : (x1,y1): ({},{})\t (x2,y2): ({},{})\n".format(x1,y1,x2,y2))
        print("Scores : {}\n".format(scores))
        # === Image exporting
        # output_frame = frame[x1:x2, y1:y2]
        # cv2.imwrite(output_file, output_frame)

        # oneD_array = frame_np_expanded.reshape(-1)
        # np.savetxt(input_path.replace('.jpg', '.txt'), oneD_array, fmt="%.8f")
        # np.savetxt(input_path.replace('.jpg', '_output.txt'), boxes[0], fmt="%.8f")


    sess.close()


def drawDetection(image,result,colors=None,cost=None):
    if image is None:
        return image
    height,width,c=image.shape
    show=image.copy()
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        if colors is None:
            cv2.putText(show,LABEL_NAME,(xmin,ymin), cv2.FONT_ITALIC,1,(0,0,255))
            cv2.rectangle(show,(xmin,ymin),(xmax,ymax),(255,0,0),3)
        else:
            color=colors[int(round(item[4]))]
            color=[c *256 for c in color]
            cv2.putText(show,LABEL_NAME,(xmin,ymin), cv2.FONT_ITALIC,1,color)
            cv2.rectangle(show,(xmin,ymin),(xmax,ymax),color)

    if not cost is None:
        cv2.putText(show,cost,(0,40),3,1,(0,0,255))

    return show   


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

