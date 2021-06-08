import tensorflow as tf
import argparse
import numpy as np
import os
from enum import Enum, unique

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.python.tools import optimize_for_inference_lib

'''
Reference:
tensorflow/tensorflow/tools/graph_transforms/python/transform_graph_test.py 
'''

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_pb', type=str, 
                    default='None', help='PB path.')
# parser.add_argument('--output_pb', type=str, 
#                     default='None', help='Output')

args = parser.parse_args()

'''
flags = tf.app.flags 
flags.DEFINE_string(
    'train_data_path',
    # '/datasets/data1/faces_ms1m_112x112/intermediate/shuffle_faces_ms1m_112x112.record', #gtx
    '/datasets/t1/data/faces_ms1m_112x112/intermediate/shuffle_faces_ms1m_112x112.record', #rtx
    'Input Training dataset path'
    )
FLAGS = flags.FLAGS
'''
# tensorflow/tensorflow/core/framework/types.proto  
@unique
class DataType(Enum):
# Not a legal value for DataType.  Used to indicate a DataType field
# has not been set.
    DT_INVALID = 0;

  # Data types that all computation devices are expected to be
  # capable to support.
    DT_FLOAT = 1;
    DT_DOUBLE = 2;
    DT_INT32 = 3;
    DT_UINT8 = 4;
    DT_INT16 = 5;
    DT_INT8 = 6;
    DT_STRING = 7;
    DT_COMPLEX64 = 8;  # Single-precision complex
    DT_INT64 = 9;
    DT_BOOL = 10;
    DT_QINT8 = 11;     # Quantized int8
    DT_QUINT8 = 12;    # Quantized uint8
    DT_QINT32 = 13;    # Quantized int32
    DT_BFLOAT16 = 14;  # Float32 truncated to 16 bits.  Only for cast ops.
    DT_QINT16 = 15;    # Quantized int16
    DT_QUINT16 = 16;   # Quantized uint16
    DT_UINT16 = 17;
    DT_COMPLEX128 = 18;  # Double-precision complex
    DT_HALF = 19;
    DT_RESOURCE = 20;
    DT_VARIANT = 21;  # Arbitrary C++ data types
    DT_UINT32 = 22;
    DT_UINT64 = 23;


def main():
    pb_graph = tf.Graph()
    output_path = '/'.join(args.input_pb.split('/')[:-1]) + '/'
    output_pb = args.input_pb.split('/')[-1].replace('.pb', '_init.pb')
    with pb_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.input_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            print('='*10, 'Check out the input placeholders:', '='*10)
            nodes = [n.name + ' => ' +  n.op for n in od_graph_def.node if n.op in ('Placeholder')]
            for node in nodes:
                print(node)
            tf.import_graph_def(od_graph_def, name='')


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with pb_graph.as_default():
    new_model = tf.GraphDef()

    with tf.Session(graph=pb_graph, config=config) as sess:
        # === 1. Create and Add new node ===
        '''
        for n in sess.graph_def.node:
        # New constant to add
            new_value = np.array((np.ones([3,3,3,8])) * 0.001, dtype=np.float32)
        # Make new graph node
            tensor_content = new_value.tobytes()
            dt = tf.as_dtype(new_value.dtype).as_datatype_enum
            tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=s) for s in new_value.shape])
            tensor_proto = TensorProto(tensor_content=tensor_content,
                                       tensor_shape=tensor_shape,
                                       dtype=dt)
            node = tf.NodeDef(name=n.name, op=n.op,
                              attr={'value': tf.AttrValue(tensor=tensor_proto),
                                    'dtype': tf.AttrValue(type=dt)})
        # Add new node
            new_model.node.extend([node])
        '''

        # === 2. Change Specific op ===
        '''
        for n in sess.graph_def.node:
            if n.op == 'Sigmoid':
                nn = new_model.node.add()
                nn.op = 'Tanh'
                nn.name = n.name
                for i in n.input:
                    nn.input.extend([i])
            else:
                nn = new_model.node.add()
                nn.CopyFrom(n)
        '''

        # === 3. Modify tensor(parameters) value ===
            # Get tensor detail if tensor content exist
        for n in sess.graph_def.node:
            if n.attr["value"].tensor.tensor_content:
                tensor_bytes = n.attr["value"].tensor.tensor_content
                tensor_dtype = n.attr["value"].tensor.dtype
                tensor_shape = [x.size for x in n.attr["value"].tensor.tensor_shape.dim]
                tensor_array = tf.decode_raw(tensor_bytes, tensor_dtype)
                tensor_array = tf.reshape(tensor_array, tensor_shape)
                npdtype = tensor_array.dtype.name
            # Create new tensor value
                # new_value = np.array((np.ones(tensor_shape)) * 0.1, dtype=np.dtype(npdtype))
                new_value = np.array((np.random.random(tensor_shape) + 0.01) * 0.001, dtype=np.dtype(npdtype))
                raw_value = np.frombuffer(n.attr["value"].tensor.tensor_content, dtype=np.dtype(npdtype))
                if npdtype == 'float32':
                    # np.random.shuffle(raw_value)
                    # tensor_content = raw_value.tobytes()
                    tensor_content = new_value.tobytes()
                    pass
                else:
                    tensor_content = raw_value.tobytes() 
                # print('raw shape: {},\t new shape: {}'.format(temp.shape, new_value.shape))
            # Update value
                nn = new_model.node.add()
                nn.CopyFrom(n)
                nn.attr["value"].tensor.tensor_content = tensor_content
            # Clone node
            else:
                nn = new_model.node.add()
                nn.CopyFrom(n)
                if not nn.attr["value"].tensor.dtype:
                    del nn.attr["value"] # trivial value induce inference error
        #========================
        # === Save pb
        print('Saving .pb file...')
        with tf.gfile.FastGFile(output_path+output_pb, 'w') as f:
            f.write(new_model.SerializeToString())

if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

'''
old_gd = tf.get_default_graph().as_graph_def()
old_nodes = old_gd.node  # old nodes from graph

nodes_after_modify = []
for node in old_nodes:
   new_node = node_def_pb2.NodeDef()  # 產生新節點
   new_node.CopyFrom(node)  # 拷貝舊節點資訊到新節點
   input_before_removal = node.input  # 把舊節點的inputs暫存起來
   if new_node.name == target_node_name:  # 如果節點是第一個con2D
       del new_node.input[:]  # 就把該inputs全部去除
       for input_name in input_before_removal:  # 然後再for跑一次剛剛刪除的inputs
           if input_name == old_input_name:  # inputs中若有舊input
               new_node.input.append(mul.op.name)  # 指到新input
           else:
               new_node.input.append(input_name)  # 不是的話，維持原先的input
   nodes_after_modify.append(new_node)  # 將新節點存到list

new_gd = graph_pb2.GraphDef()  # 產生新graph def
new_gd.node.extend(nodes_after_modify)  # 在新graph def中生成那些新節點後return
return new_gd
'''