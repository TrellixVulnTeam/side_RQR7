# import the necessary packages
from __future__ import print_function
# from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path, PurePath
# import imutils
import time
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="output dir")
ap.add_argument("-i", "--input", required=True,
	help="input dir")
ap.add_argument("-m", "--model", required=True,
	help="pb model")
ap.add_argument("--height", type=int, default=None,
	help="resize height")
ap.add_argument("--width", type=int, default=None,
	help="resize width")
args = vars(ap.parse_args())


def ImgFD():
	#= Setting pb model =#
	print("[INFO] reading model...")
	if args["model"] is None: 
	    raise ValueError('!!! Error model !!!')
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	    od_graph_def = tf.GraphDef()
	    with tf.gfile.GFile(args["model"], 'rb') as fid:
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
	FD_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	FD_scores = detection_graph.get_tensor_by_name('detection_scores:0')


	# initialize the video stream and allow the camera
	# sensor to warmup
	(h, w) = (args["height"], args["width"])
	print("[INFO] reading images...")

	match_type = '*'+'.'+'jpg'

	input_list = []
	output_list = []
	output_dir = Path(args["output"])
	if not output_dir.is_dir():
		Path(args["output"]).mkdir()
		output_dir = Path(args["output"])

	for f in Path(args["input"]).glob('**/*'):
		if PurePath(f).match(match_type):
			file_path = str(f)
			input_list.append(file_path)
			output_file = output_dir.joinpath(file_path.split('/')[-1])
			output_list.append(str(output_file))

	print("[INFO] translating images...")
	print("[INFO] only crop 1 person...")
	for i in range(len(input_list)):
		frame = cv2.imread(input_list[i])
		if frame is None: 
		    raise ValueError('!!! Error opening image !!!')
		image_height, image_width, channels = frame.shape
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_np_expanded = np.expand_dims(frame_rgb, axis=0)
		boxes,scores = sess.run([FD_boxes, FD_scores],
		                        feed_dict={i_tensor: frame_np_expanded}) # execution

		max_i = np.argmax(scores)
		box_coord = boxes[0][max_i] * np.array([image_width,image_height,image_width,image_height])
		x1, y1, x2, y2 = box_coord.astype('int')
		output_frame = frame[x1:x2, y1:y2]

		# frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		resize_frame = cv2.resize(output_frame, (w, h), interpolation=cv2.INTER_AREA)
		cv2.imwrite(output_list[i], resize_frame)


def ImgResize():
	# initialize the video stream and allow the camera
	# sensor to warmup
	(h, w) = (args["height"], args["width"])
	print("[INFO] reading images...")

	match_type = '*'+'.'+'jpg'

	input_list = []
	output_list = []
	output_dir = Path(args["output"])
	if not output_dir.is_dir():
		Path(args["output"]).mkdir()
		output_dir = Path(args["output"])

	for f in Path(args["input"]).glob('**/*'):
		if PurePath(f).match(match_type):
			file_path = str(f)
			input_list.append(file_path)
			output_file = output_dir.joinpath(file_path.split('/')[-1])
			output_list.append(str(output_file))

	print("[INFO] translating images...")
	for i in range(len(input_list)):
		frame = cv2.imread(input_list[i])
		if frame is None: 
		    raise ValueError('!!! Error opening image !!!')
		frame = cv2.resize(frame, (w, h))
		# frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		resize_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
		cv2.imwrite(output_list[i], resize_frame)

	# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# if args["fps"] is None:
	# 	args["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
	# 	pass

	# cv2.imwrite(output_file, image_live_np)
	# print("[INFO] cleaning up...")
	# cv2.destroyAllWindows()
	# cap.release()
	# writer.release()

def main():

    ImgFD()

    print('Translate finished !')


if __name__ == '__main__':
   main()