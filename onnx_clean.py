import onnx

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--onnx_path', type=str, 
					default='None', help='Onnx path')
parser.add_argument('--input_shape', type=str, 
					default='', help='Input shape C,H,W')

args = parser.parse_args()

if args.onnx_path is 'None':
	raise ValueError(
					 '{} is None, Please enter onnx_path'.format(args.onnx_path))
	pass

input_shape = [int(i) for i in args.input_shape.split(',')]
model = onnx.load(args.onnx_path)

print(model.graph.input[0].type.tensor_type.shape.dim)

model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = input_shape[0]
model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = input_shape[1]
model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = input_shape[2]
print(model.graph.input[0].type.tensor_type.shape.dim)
input_name = model.graph.input[0].name
del model.graph.node[0]
model.graph.node[0].input[0] = input_name

save_onnx_path = args.onnx_path.replace('.onnx', '_new.onnx')
onnx.save(model, save_onnx_path)