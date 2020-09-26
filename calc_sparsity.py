import tensorflow as tf
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_pb', type=str, 
                    default='None', help='PB path.')
parser.add_argument('--tensor_name', type=str, 
                    default='None', help='Tensor name, ex weight')

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

def main():

    pb_graph = tf.Graph()
    with pb_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.input_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with pb_graph.as_default():
        with tf.Session(graph=pb_graph, config=config) as sess:

        # ===Get by Name===
            # _tensor = pb_graph.get_tensor_by_name(args.tensor_name+':0')
            # print(sess.run(_tensor))
        # ===Get collection by op that include the tensor_name===
            tensor_collections = [i.name for i in pb_graph.get_operations() if args.tensor_name in i.name]
            for x in tensor_collections:
                _tensor = pb_graph.get_tensor_by_name(x+':0')
                print("-"*100)
                tensor = sess.run(_tensor)
                print(tensor.shape)
                count = 0
                prune_filters = []
                size = 1
                for i in tensor.shape:
                    size = i*size
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[1]):
                        for k in range(tensor.shape[2]):
                            for l in range(tensor.shape[3]):
                                if tensor[i,j,k,l] == 0:
                                    count += 1
                print('Sparsity = {0} \n'.format(count/size))
                
                # print(x)
                # if tensor.shape[0] == 3:
                #     for filter_i in range(tensor.shape[2]):
                #         _abs = np.abs(tensor[:,:,filter_i,0])
                #         _sum = np.sum(_abs)
                #         if _sum == 0:
                #             count += 1
                #             prune_filters.append(filter_i)
                #     print('Sparsity = {0} \n'.format(count/tensor.shape[2]))
                # else:
                #     for filter_i in range(tensor.shape[3]):
                #         _abs = np.abs(tensor[:,:,:,filter_i])
                #         _sum = np.sum(_abs)
                #         if _sum == 0:
                #             count += 1
                #             prune_filters.append(filter_i)
                #     print('Sparsity = {0} \n'.format(count/tensor.shape[3]))
                # print(prune_filters)                

if __name__ == '__main__':
   main()
   # FLAGS, unparsed = parser.parse_known_args()
   # tf.app.run()

