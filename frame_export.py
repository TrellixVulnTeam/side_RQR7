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
from input_pipeline import Pipeline
import os

flags = tf.app.flags
flags.DEFINE_string(
    'model',
    'None',
    'Model PATH'
)
flags.DEFINE_string(
    'data_type',
    'video',
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
    'model_type',
    'MobilePose',
    'Model architecture in [MobilePose, FPMobilePose, SEMobilePose, sppe]'
)
flags.DEFINE_string(
    'loss_fn',
    'MSE',
    'Loss function in [MSE, softmax, center, focal, inv_focal, arcface]'
)
flags.DEFINE_boolean(
    'data_augmentation',
    False,
    'Add data augmentation to preprocess'
)
flags.DEFINE_integer(
    'number_keypoints',
    17,
    'Number of keypoints in [17, 12]'
)
FLAGS = flags.FLAGS

def TFlitePerformance(tflite_path, data_path, pipeline_param):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Getting input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pip = Pipeline()

    input_data = pip.eval_data_pipeline(
        data_path,
        params = pipeline_param,
        batch_size = 1
    )

    start = timeit.default_timer() # start timing

    data_count = 40
    output_data = []
    input_video = []
    with tf.Session() as sess:
        while data_count > 0:
            try:
                input_temp = sess.run(input_data[0])
                input_video.append(input_temp) # for OpenCV video background
                # 針對不同的tflite，需要不同的input dtype,可能是 float32 或 uint8
                # input_temp = (input_temp + 1)*255/2
                # input_temp = input_temp.astype('uint8')

                interpreter.set_tensor(input_details[0]['index'], input_temp )
                interpreter.invoke()

                output_temp = interpreter.get_tensor(output_details[0]['index'])
                output_data.append(output_temp) # for OpenCV PE mark
                data_count -= 1
                # print(output_data)
            except tf.errors.OutOfRangeError:
                break

    input_video = np.array(input_video)
    output_data = np.array(output_data)
    CV_performance_image(input_video[:,0,...], output_data, output_video_path, run = True)
    print('Performmance Process Finished.')
    print(f'cost time:{(timeit.default_timer() - start)} sec')
    # print(output_data.dtype)


def Inference_tflite(input_path, input_type, output_path, model_path):

    #= Parameter setting =#
    ti1 = tt.time()
    tt1 = 0
    c1 = 0
    c2 = 0
    frame_rate = 0
    # image_width = 1080
    # image_height = 1440
    image_width = 314
    image_height = 353
    image_crop_w = 256
    image_crop_h = 320
    need_rotation = False

    if output_path == 'None':
        output_path = '/'.join(input_path.split('/')[:-1])
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = output_path + '/{}'.format(input_path.split('/')[-1])
    
    #= Generating corresponding output path =#
    if input_type == 'video':
        cap = cv2.VideoCapture(input_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        output_file = output_path.replace('.mp4', '_tflite.mp4')
        # output_file = output_path.replace('.mp4', '_test.jpg')
        # print(output_file)

        # 使用 XVID 編碼，FPS 值為 20.0，解析度為 image_width x image_height
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(output_file, fourcc, 20.0, (image_width, image_height))

    elif input_type == 'image' :
        pic = cv2.imread(input_path)
        if pic is 'None':
            print("Error image path")
        # pic = cv2.resize(pic, (image_width, image_height))
        output_file = output_path.replace('.jpg', '_tflite.jpg')
    print('===Output File===\n',output_file)
    #= Setting tflite model =#
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Getting input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Getting specific tensor
    tensor_details = interpreter.get_tensor_details()
    '''
    # ============ Show Tensor details ============
    for x in range(len(tensor_details)):
        # Print all tensor index:
        print('{} : {}'.format(tensor_details[x]['name'], tensor_details[x]['index']))
        if('depth_to_space' in tensor_details[x]['name'] ):
            print('{} : {} : {}'.format(tensor_details[x]['name'], tensor_details[x]['index'], tensor_details[x]['shape']))
        if( 'Relu' in tensor_details[x]['name']):
            print('{} : {} : {}'.format(tensor_details[x]['name'], tensor_details[x]['index'], tensor_details[x]['shape']))
        pass
    sys.exit(1)
    '''

    if input_type == 'video':
        while cap.isOpened():
            # Getting frame
            ret, frame_live = cap.read()
            # first_img = np.zeros((H, W, 3))
            if ret:
                image_live_np, cropped = operation_crop(frame_live, image_crop_w, image_crop_h, 
                                         image_width, image_height, need_rotation)                
                #=== tflite operation ===
                #=== convert to input data type
                input_frame = cropped.astype('float32')
                interpreter.set_tensor(input_details[0]['index'], input_frame)
                t1 = tt.time()
                interpreter.invoke()
                hms = interpreter.get_tensor(output_details[0]['index'])

                it1 = tt.time()-t1
                tt1 += it1
                c1 += 1
                if (tt.time()-ti1) >= 1:
                    ti1 = tt.time()
                    frame_rate = 1 / (tt1 / c1)
                    c1 = 0
                    tt1 = 0

                # image_live_np = frame_live.copy()
                cv2.putText(image_live_np, '{:.1f}'.format(frame_rate), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

                # Reverting frame
                operation_plot(image_live_np, hms, image_crop_w, image_crop_h)
                # Video exporting
                video_out.write(image_live_np)
            else:
                break
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame_live = pic
        image_live_np, cropped = operation_crop(frame_live, image_crop_w, image_crop_h, 
                                 image_width, image_height, need_rotation)                
        #=== tflite operation ===
        #=== convert to input data type
        input_frame = cropped.astype('float32')
        # Get input and output ID for opencl kernal operation node id
        # print(input_details[0]['index'])
        # print(output_details[0]['index'])
        
        interpreter.set_tensor(input_details[0]['index'], input_frame)
        t1 = tt.time()
        interpreter.invoke()
        hms = interpreter.get_tensor(output_details[0]['index'])

        '''
        # ============= Export In/ Output 1D array for verify ==============   
        inter_out = interpreter.get_tensor(193)
        print('Inter output shape{0}'.format(inter_out.shape))
        oneD_array = input_frame.reshape(-1)
        np.savetxt('/workspace/image2.txt', oneD_array, fmt="%.8f,")
        oneD_array = hms.reshape(-1)
        scale = (oneD_array+1)*255/2 - 127
        # np.savetxt('/workspace/PE_SEN_opt_tflite_out_img1.txt', oneD_array, fmt="%.8f")
        np.savetxt('/workspace/PE_SEN_opt_tflite_float_out_img1.txt', oneD_array, fmt="%5d")
        sys.exit(1)
        '''

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
        cv2.putText(image_live_np, '{:.1f}'.format(frame_rate), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

        # Reverting frame
        operation_plot(image_live_np, hms, image_crop_w, image_crop_h)
        # Image exporting
        # output_frame = (image_processed+1) * 255 /2
        cv2.imwrite(output_file, image_live_np)


def Inference_pb(input_path, input_type, output_path, model_path):

    #= Parameter setting =#
    ti1 = tt.time()
    tt1 = 0
    c1 = 0
    c2 = 0
    frame_rate = 0
    # image_width = 1080
    # image_height = 1440
    image_width = 314
    image_height = 353
    image_crop_w = 256
    image_crop_h = 320
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

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

        output_file = output_path.replace('.mp4', '_pb.mp4')

        # 使用 XVID 編碼，FPS 值為 20.0，解析度為 image_width x image_height
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(output_file, fourcc, 20.0, (image_width, image_height))

    elif input_type == 'image':
        pic = cv2.imread(input_path)
        # pic = cv2.resize(pic, (image_width, image_height))
        output_file = output_path.replace('.jpg', '_pb.jpg')

    #= Setting pb model =#
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    sess = tf.Session(graph=detection_graph, config=config)
    crp_img = detection_graph.get_tensor_by_name('cropped_image:0')
    hms_tensor = detection_graph.get_tensor_by_name('output/BiasAdd:0')

    if input_type == 'video':
        while cap.isOpened():
            # Getting frame
            ret, frame_live = cap.read()
            # first_img = np.zeros((H, W, 3))
            if ret:
                image_live_np, cropped = operation_crop(frame_live, image_crop_w, image_crop_h, 
                                         image_width, image_height, need_rotation)                
                #=== pb operation ===
                t1 = tt.time()
                hms = sess.run(hms_tensor, feed_dict={crp_img: cropped}) # execution

                it1 = tt.time()-t1
                tt1 += it1
                c1 += 1
                if (tt.time()-ti1) >= 1:
                    ti1 = tt.time()
                    frame_rate = 1 / (tt1 / c1)
                    c1 = 0
                    tt1 = 0

                # image_live_np = frame_live.copy()
                cv2.putText(image_live_np, '{:.1f}'.format(frame_rate), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

                # Reverting frame
                operation_plot(image_live_np, hms, image_crop_w, image_crop_h)
                # Video exporting
                video_out.write(image_live_np)
            else:
                break
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame_live = pic
        image_live_np, cropped = operation_crop(frame_live, image_crop_w, image_crop_h, 
                                 image_width, image_height, need_rotation)                
        #=== pb operation ===
        t1 = tt.time()
        hms = sess.run(hms_tensor, feed_dict={crp_img: cropped}) # execution

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
        cv2.putText(image_live_np, '{:.1f}'.format(frame_rate), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

        # Reverting frame
        operation_plot(image_live_np, hms, image_crop_w, image_crop_h)
        # Image exporting
        # output_frame = (image_processed+1) * 255 /2
        cv2.imwrite(output_file, image_live_np)


    sess.close()



# Reverting the position of mark and label the positions on frame 
def operation_plot(frame, humanmarks, crop_W, crop_H):
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
def operation_crop(frame, crop_W, crop_H, image_W, image_H, rotation=False):
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
    if (model_type == 'pb') or (model_type == 'tflite') :
        print('{} supported.'.format(model_name))
        if FLAGS.data_type == 'video' or  FLAGS.data_type == 'image':
            print('Video or Image path:\n {}'.format(FLAGS.data_path))
        else:
            print('Type of {} not supported.'.format(FLAGS.data_type))
    else:
        print('Model name {} not supported.'.format(model_name))
        return 0

    # pipeline_param = {
    #     'model_arch': FLAGS.model_type,
    #     'do_data_augmentation': FLAGS.data_augmentation,
    #     'loss_fn': FLAGS.loss_fn,
    #     'number_keypoints': FLAGS.number_keypoints
    # }

    # Inference Execution
    if model_type == 'pb':
        Inference_pb(FLAGS.data_path, FLAGS.data_type, FLAGS.output_path, FLAGS.model)
    elif model_type == 'tflite':
        Inference_tflite(FLAGS.data_path, FLAGS.data_type, FLAGS.output_path, FLAGS.model)

    print('Interprete finished !')


if __name__ == '__main__':
    tf.app.run()

