from tensorflow.python.framework import tensor_util
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from numpy import prod, sum
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_pb', type=str, 
                    default=None, help='PB path.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flops = 0
GRAPH_PB_PATH = args.input_pb #path to your .pb file
with tf.Session() as sess:
	print("load graph")
	with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
		graph_def = tf.GraphDef()
		graph = tf.get_default_graph()
		graph_def.ParseFromString(f.read())
				
		for node in graph_def.node :
		    if node.op == 'RefSwitch':
			    node.op = 'Switch'
			    for index in range(len(node.input)):
				    if 'moving_' in node.input[index]:
				    	node.input[index] = node.input[index] + '/read'
		    elif node.op == 'AssignSub':
			    node.op = 'Sub'
			    if 'use_locking' in node.attr: 
				    del node.attr['use_locking']
		
		tf.import_graph_def(graph_def, name='')
		num_layers = 0
		print('OP name \t\t Output shape \t Input shape:')
		for op in graph.get_operations():
			if(op.type == "Conv2D" or op.type == "DepthwiseConv2dNative"):
				flops += op.outputs[0].shape[1] * op.outputs[0].shape[2] * prod(op.inputs[1].shape)
				#flops += op.outputs[0].shape[1] * op.outputs[0].shape[2] * prod(op.inputs[1].shape) + prod(prod(op.outputs[0].shape))
				print(op.name, op.outputs[0].shape, op.inputs[0].shape)
				num_layers += 1
		print("Total layers: ", num_layers)
		print("FLOPs: ", flops)