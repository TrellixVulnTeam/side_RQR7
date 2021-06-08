import tensorflow as tf
import argparse
import numpy as np
import os
from enum import Enum, unique

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

'''
Reference:
tensorflow/tensorflow/tools/graph_transforms/python/transform_graph_test.py 
'''

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_pb', type=str, 
                    default=None, help='PB path.')
parser.add_argument('--init', type=bool, 
                    default=False, help='Initialized option')
parser.add_argument('--delete', type=bool, 
                    default=False, help='Delete Ops option')
parser.add_argument('--delete_name', type=str, 
                    default=None, help='Delete Op')
parser.add_argument('--remove', type=bool, 
                    default=False, help='Remove Node option')
parser.add_argument('--remove_name', type=str, 
                    default=None, help='Remove node')
parser.add_argument('--replace', type=bool, 
                    default=False, help='Replace Node option')
parser.add_argument('--replace_name', type=str, 
                    default=None, help='Replace node')

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

# Only delete simple op, like pool.
def DeleteOps(graph_def, delete_op_name):
    # Delete nodes
    nodes_after_modify = []
    inputs_map = {}
    for node in graph_def.node:
        if delete_op_name in node.name:
            inputs_map[node.name]= node.input
            print('Drop', node.name)
        else:
            nodes_after_modify.append(node)

    new_graph_def = tf.GraphDef()
    new_graph_def.node.extend(nodes_after_modify)

    # Delete references to deleted nodes
    for node in new_graph_def.node:
        inp_names = []
        for inp in node.input:
            if delete_op_name in inp:
                inp_names.extend(inputs_map[inp]) # inputs_map[inp] is list not element
                pass
            else:
                inp_names.append(inp)

        del node.input[:]
        print(inp_names)
        node.input.extend(inp_names)

    return new_graph_def


#  remove node and connect its input to follower
# A-input b->B-input c->C-input d->D and going to remove say node B 
# then we should not just remove input c but replace it with input b
def RemoveNode(graph_def, remove_node_name, specific_input=None):
    nodes_after_modify = []
    for node in graph_def.node:
        if node.name == remove_node_name:
            assert(len(node.input) != 0), "Node input dose not exist"
            # Avoid specific input
            assert(specific_input not in node.input),"Node input could not be removed"
            # node_input = [name for name in node.input]
            input_of_removed_node = node.input if len(node.input) else ''
            print("Removing {} and using its input {}".format(node.name, 
                   input_of_removed_node))
            continue
        nodes_after_modify.append(node)
    
    # modify inputs where required
    # removed name must be replaced with input from removed node
    for node in nodes_after_modify:
        inp_names = []
        replace = False
        for inp in node.input:
            if inp == remove_node_name:
                inp_names.extend(input_of_removed_node)
                print("For node {} replacing input {} with {}".format(node.name, inp, input_of_removed_node))
                replace = True
            else:
                inp_names.append(inp)
        if replace:
            del node.input[:]
            node.input.extend(inp_names)
    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes_after_modify)
    return mod_graph_def


def add_preprocessing(target_node_name, old_input_name):
    # create preprocessing node
    new_input_node = tf.placeholder(shape=[None, 128, 128, 3], dtype=tf.float32, name='new_input_node')
    with tf.variable_scope('pre_processing'):
        sub = tf.subtract(new_input_node, 127.5)
        mul = tf.multiply(sub, 0.0078125, name='out')

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

    new_gd = graph_pb2.GraphDef()
    new_gd.node.extend(nodes_after_modify)
    return new_gd


def remove_training_node(is_training_node, false_node_name):
    false_node = tf.constant(False, dtype=tf.bool, shape=(), name=false_node_name)

    old_gd = tf.get_default_graph().as_graph_def()
    old_nodes = old_gd.node  # old nodes from graph

    nodes_after_modify = []
    for node in old_nodes:
        new_node = node_def_pb2.NodeDef()  # 產生新節點
        new_node.CopyFrom(node)  # 拷貝舊節點資訊到新節點
        input_before_removal = node.input  # 把舊節點的inputs暫存起來
        del new_node.input[:]  # 就把該inputs全部去除
        for full_input_name in input_before_removal:  # 然後再for跑一次剛剛刪除的inputs
            if full_input_name == is_training_node:  # inputs中若有training_node
                new_node.input.append(false_node.op.name)  # 改塞false_node給它
            else:
                new_node.input.append(full_input_name)  # 不是的話，維持原先的input
        nodes_after_modify.append(new_node)  # 將新節點存到list

    new_gd = graph_pb2.GraphDef()  # 產生新graph def
    new_gd.node.extend(nodes_after_modify)  # 在新graph def中生成那些新節點後return
    return new_gd

def ReplaceInput(graph_def, replace_node_name):
    # create new input node
    new_input_node = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='new_input')

    nodes_after_modify = []
    # Creat new node route
    for node in graph_def.node:
        if node.name == replace_node_name:
            pass # pass replace node
        else:
            nodes_after_modify.append(node)
    # Modify inputs
    for node in nodes_after_modify:
        if node.input == replace_node_name:
            print(node.input)
            del node.input[:]
            node.input.append(new_input_node.op.name)

    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes_after_modify)
    return mod_graph_def

def ReplaceToFloat(graph_def, replace_node_name):
    # create new input node
    new_input_node = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='new_input')
    remove_node = 'ToFloat'

    nodes_after_modify = []
    # Creat new node route
    for node in graph_def.node:
        if node.name == replace_node_name or node.name == remove_node:
            print('Pass node: {}'.format(node.name))
            print('Input node: {}'.format(node.input))
            pass # pass replace node
        else:
            nodes_after_modify.append(node)
    # Modify inputs
    print('-'*10, 'modify input', '-'*10)
    for node in nodes_after_modify:
        if remove_node in node.input:
            print('Before input node: {}'.format(node.input))
            del node.input[:]
            node.input.append(new_input_node.op.name)
            print('After input node: {}'.format(node.input))
        if replace_node_name in node.input:
            del node.input[:]
    print('-'*10, 'build graph', '-'*10)

    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes_after_modify)
    return mod_graph_def


def RemoveBefore(graph_def, junction_node):

    nodes_after_modify = []
    junction = False
    # Creat new node route
    for node in graph_def.node:
        if node.name == junction_node:
            print('Junction node: {}'.format(node.name))
            nodes_after_modify.append(node)
            junction = True
        if junction:
            nodes_after_modify.append(node)
    # Modify inputs
    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes_after_modify)
    return mod_graph_def


def InitVars(graph_def):
    new_graph_def = tf.GraphDef()
    # === 3. Modify tensor(parameters) value ===
        # Get tensor detail if tensor content exist
    for n in graph_def.node:
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
            # if 'weight' in n.name:
            #     raw_value = raw_value * 1.08
            #     tensor_content = raw_value.tobytes()
            # elif 'moving' in n.name:
            #     raw_value = raw_value * 0.95
            #     tensor_content = raw_value.tobytes()
            # elif 'beta' in n.name or 'gamma' in n.name:
            #     raw_value = raw_value * 1.01
            #     tensor_content = raw_value.tobytes()
            else:
                tensor_content = raw_value.tobytes() 
            # print('raw shape: {},\t new shape: {}'.format(temp.shape, new_value.shape))
        # Update value
            nn = new_graph_def.node.add()
            nn.CopyFrom(n)
            nn.attr["value"].tensor.tensor_content = tensor_content
        # Clone node
        else:
            nn = new_graph_def.node.add()
            nn.CopyFrom(n)
            if not nn.attr["value"].tensor.dtype:
                del nn.attr["value"] # trivial value induce inference error
    return new_graph_def

def update_graph(graph_def):
    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')

def main():
    output_path = '/'.join(args.input_pb.split('/')[:-1]) + '/'

    pb_graph = tf.GraphDef()
    with tf.gfile.GFile(args.input_pb, 'rb') as fid:
        serialized_graph = fid.read()
        pb_graph.ParseFromString(serialized_graph)
        print('='*10, 'Check out the input placeholders:', '='*10)
        nodes = [n.name + ' => ' +  n.op for n in pb_graph.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)
        tf.import_graph_def(pb_graph, name='')


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.init:
        new_model = InitVars(pb_graph)
        update_graph(new_model)
        output_pb = args.input_pb.split('/')[-1].replace('.pb', '_init.pb')
        pass
    if args.delete:
        new_model = DeleteOps(pb_graph, args.delete_name)
        update_graph(new_model)
        output_pb = args.input_pb.split('/')[-1].replace('.pb', '_delete.pb')
        pass 
    if args.remove:
        # new_model = RemoveNode(pb_graph, args.remove_name)
        new_model = RemoveBefore(pb_graph, args.remove_name)
        # update_graph(new_model)
        output_pb = args.input_pb.split('/')[-1].replace('.pb', '_remove.pb')
        pass 
    if args.replace:
        new_model = ReplaceToFloat(pb_graph, args.replace_name)
        # update_graph(new_model)
        output_pb = args.input_pb.split('/')[-1].replace('.pb', '_replace.pb')
        with tf.Session(config=config) as sess:
            final_gd = tf.graph_util.convert_variables_to_constants(
                        sess, new_model, ['detection_boxes','detection_scores','num_detections','detection_classes'])

        pass 
    #========================
    # === Save pb
    print('Saving .pb file...')
    with tf.gfile.FastGFile(output_path+output_pb, 'w') as f:
        f.write(new_model.SerializeToString())
    exit()
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

        # final_gd = tf.graph_util.convert_variables_to_constants(
        #             sess, tf.get_default_graph().as_graph_def(), ['final_dense/MatMul'])
        if args.replace:
            # new_model = ReplaceInput(sess, args.replace_name)
            # new_model = delete_ops_from_graph(sess, args.replace_name)
            new_model = RemoveNode(sess, args.replace_name, 'image_tensor')
            # update_graph(new_model)
            new_model = ReplaceInput(new_model, 'image_tensor')
            new_model = tf.graph_util.convert_variables_to_constants(
                        sess, new_model, ['detection_boxes','detection_scores','num_detections','detection_classes'])
            output_pb = args.input_pb.split('/')[-1].replace('.pb', '_replace.pb')
            pass        

        if args.init:
            new_model = InitVars(sess.graph_def)
            output_pb = args.input_pb.split('/')[-1].replace('.pb', '_init.pb')
            pass
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