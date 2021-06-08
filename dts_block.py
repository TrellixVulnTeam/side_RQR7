import tensorflow as tf
import numpy as np
import os

def PixelShuffle(I, scope='PixelShuffle'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        I = tf.layers.conv2d(I, 1024, 2)
        ps = tf.depth_to_space(I, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')

        sp = tf.space_to_depth(ps, block_size=128, data_format='NHWC')

        ps = tf.depth_to_space(sp, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')

        sp = tf.space_to_depth(ps, block_size=128, data_format='NHWC')

        ps = tf.space_to_depth(sp, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')
        ps = tf.depth_to_space(ps, block_size=2, data_format='NHWC')

    return ps

def Build_graph(sess):
    print('Build graph...')

    inputs = tf.placeholder(tf.float32,
                            shape=(1, 33, 33, 1024),
                            name='input')
    output = PixelShuffle(inputs)
    logits = tf.identity(output, name='output')
    sess = tf.Session()
    sess.run(tf.initializers.global_variables())
    # sess.run(end_points['heat_map'], feed_dict={inputs: np.zeros((1, 320, 256, 3))})
    saver = tf.train.Saver() #可指定需要儲存的tensor，不指定則全部儲存

    sess.run(tf.global_variables_initializer())
    #建立儲存模型的資料夾
    if not os.path.exists('my_model'):
        os.mkdir('./my_model')
        saver.save(sess, './my_model/my_test_model')
    #可通過設定saver.save()的引數指定儲存哪一步的模型
    saver.save(sess, './my_model/my_test_model', global_step=1000) #儲存1000步的模型


def save_pb(sess, graph):
    graph_folder = './my_model'
    graph_name = 'test.pb'
    # graph_, _ = ge.detach_inputs(graph)
    # graph_ = ge.make_placeholder(graph, scope='input', dtype=tf.float32, shape=(1, 112, 112, 3))
    # frozen_graph = tf.graph_util.convert_variables_to_constants(
    #     sess, graph_.as_graph_def(), ['Logits/LinearConv1x1/BiasAdd'])
    # tf.train.write_graph(frozen_graph,
    #                      graph_folder,
    #                      graph_name,
    #                      False)
    
    
    with graph.as_default():
        # ge.make_placeholder_from_dtype_and_shape(scope='input', dtype=tf.float32, shape=(1, 112, 112, 3))
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['output'])
        tf.train.write_graph(frozen_graph,
                             graph_folder,
                             graph_name,
                             False)



def main(_):
    # prepare the test data
    with tf.Session() as sess:

        # Build(sess)
        # load the meta graph and weights
        saver = tf.train.import_meta_graph('./my_model/my_test_model-1000.meta')
        saver.restore(sess, './my_model/my_test_model-1000')

        # get graph
        graph = tf.get_default_graph()

        save_pb(sess, graph)


if __name__ == '__main__':
    tf.app.run()