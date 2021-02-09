import tensorflow as tf
import argparse
import numpy as np
import os
import pathlib
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pb', type=str, required=True,
                    default=None, help='PB Path.')
parser.add_argument('-j', '--jpg', type=str, required=True,
                    default=None, help='Image Path.')
parser.add_argument('-i', '--input_tensor', type=str, required=True,
                    default='input', help='Input Tensor.')
parser.add_argument('-o', '--output_tensor', type=str, required=True,
                    default=None, help='Output Tensor')
parser.add_argument('--output_dir', type=str, 
                    default=None, help='Output Dir')

args = parser.parse_args()

output_dir = args.output_dir
if not output_dir:
    output_dir = args.pb.replace('.pb','_layerout')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, mode = 0o777)
    pass
# pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

'''
python3 /workspace/side/pb_layer_output.py \
-p ./infer_test/output_8thPruned_A.pb \
-j ./infer_test/debug/test.jpg \
-i input -o SE1/mul,SE2/mul,SE3/mul,SE4/mul,DUC1/Relu,DUC2/Relu \
--output_dir ./infer_test/debug/8th_A
'''

CROP_W = 256
CROP_H = 320

def normalize8(I):
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8)

def normalize(I):
    mean = I.mean()
    var = I.max() - I.min()

    I = ((I - mean)/var)
    return I

def main():

    pb_graph = tf.Graph()
    with pb_graph.as_default():
        od_graph_def = tf.GraphDef()
        print('{} Import PB Graph... {}'.format('='*5, '='*5))
        with tf.gfile.GFile(args.pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        print('{} Image Resizing... {}'.format('='*5, '='*5))
        img = cv2.imread(args.jpg)
        HEIGHT, WIDTH, CHANNELS = img.shape
        frame = cv2.resize(img, (CROP_W, CROP_H))
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_image = cv2.resize(frame_RGB, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
        crop_image = (crop_image - 127.5) * 0.0078125 # normalize to (-1, 1)
        crop_image = np.expand_dims(crop_image, 0)


        with tf.Session() as sess:

            print('{} Feed Tensor... {}'.format('='*5, '='*5))
            input_name = args.input_tensor + ':0'
            input_T = tf.get_default_graph().get_tensor_by_name(input_name)
            output_tensors = args.output_tensor.split(',')

            for out in output_tensors:

                output_name = out + ':0'
                output_T = tf.get_default_graph().get_tensor_by_name(output_name)
                output = sess.run(output_T, feed_dict={input_T: crop_image})
                _, H, W, C = output.shape

                print('{} Saving Individual Img... {}'.format('='*5, '='*5))
                # for ch in range(C):
                #     image = output[0, :, :, ch]
                #     image = normalize8(image)
                #     heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
                #     output_file = '/'.join(output_dir.split('/')[:]) + '/' + str(ch+1) + '.jpg'
                #     print(output_file)
                #     cv2.imwrite(output_file, heatmap)
                #     pass

                # Combine all images into one pic
                print('{} Saving Merged Img... {}'.format('='*5, '='*5))
                merge_w = int(np.trunc(np.sqrt(C)))
                merge_h = int(np.ceil(C / np.trunc(np.sqrt(C))))
                image = np.zeros((merge_h*H, merge_w*W, 1), dtype=float)
                w_i = 0
                h_i = 0
                for ch in range(C):
                    w_i = int(ch % merge_w) * W
                    h_i = int(np.trunc(ch / merge_w)) * H
                    sub_img = output[0, :, :, ch]
                    # image[h_i:H, w_i:W, 0] = normalize(sub_img)
                    image[h_i:(h_i+H), w_i:(w_i+W), 0] = sub_img
                    pass
                image = normalize8(image)
                heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
                # merge_name = args.pb.split('/')[-1].replace('.pb', '_merge.jpg')
                merge_name = '_'.join(out.split('/')[:]) + '.jpg'
                output_file = '/'.join(output_dir.split('/')[:]) + '/' + merge_name
                cv2.imwrite(output_file, heatmap)

            print('{} Finished. {}'.format('='*5, '='*5))



                
if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

