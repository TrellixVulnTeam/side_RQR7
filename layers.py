import tensorflow as tf

def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.99):
    """ Assume nd [batch, N1, N2, ..., Nm, Channel] tensor"""
    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[-1]
        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, list(range(len(x.get_shape())-1)))
        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)

def _variable_on_device(name, shape, initializer, regularizer = None, trainable=True):
    dtype = 'float'
    # collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]
    with tf.device('/cpu:0'):
        var = tf.get_variable(
          name, shape, initializer=initializer, dtype=dtype, regularizer = regularizer, trainable=trainable)
    return var
def _variable_with_weight_decay(name, shape, wd, initializer, regularizer = None,  trainable=True):
    dtype = 'float'
    # collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]
    with tf.device('/cpu:0'):
        var = tf.get_variable(
          name, shape, initializer=initializer, dtype=dtype, regularizer = regularizer, trainable=trainable)
    return var

def bn_layer(layer_name, inputs, num_outputs, size, stride, padding='SAME', conv_type='Conv',freeze=False, relu=True, stddev=0.001):
    decay = 0.99
    WEIGHT_DECAY = 0.00004
    #is_training could not be defined here
    is_training = True
 
    with tf.variable_scope(layer_name) as scope:
        channels = inputs.get_shape()[3]
        kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
        mean_val   = tf.constant_initializer(0.0)
        var_val    = tf.constant_initializer(1.0)
        gamma_val  = tf.constant_initializer(1.0)
        beta_val   = tf.constant_initializer(0.0)
        
        #parameter_bn_shape : num channels
        parameter_bn_shape = conv.get_shape()[-1]
        # gamma: a trainable scale factor
        gamma = _variable_on_device('gamma', parameter_bn_shape, gamma_val, trainable=(not freeze))
        # beta: a trainable shift value
        beta  = _variable_on_device('beta', parameter_bn_shape, beta_val, trainable=(not freeze))
 
        moving_mean  = _variable_on_device('moving_mean', parameter_bn_shape, mean_val, trainable=False)
        moving_variance   = _variable_on_device('moving_variance', parameter_bn_shape, var_val, trainable=False)
        # self.model_params += [gamma, beta, moving_mean, moving_variance]
 
        # tf.nn.moments == Calculate the mean and the variance of the tensor x
        # list(range(len(conv.get_shape()) - 1)) => [0, 1, 2]
        # tf.nn.moments(tensor, axis)
        if is_training:
            mean, variance = tf.nn.moments(conv, list(range(len(conv.get_shape()) - 1)))
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, zero_debias=False)
        
            # first run update_moving_mean, update_moving_variance, then run mean, variance
            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)
            avg, var = mean_var_with_update()
        else:
            avg, var = moving_mean, moving_variance

        return tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=0.001)


def conv2d(input_, num_outputs, k_h, k_w, stride, stddev=0.02, name='conv2d', bias=False):
    # with tf.variable_scope(name):
    w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], num_outputs],
          regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
          initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, pruning.apply_mask(w, name), strides=[1, stride, stride, 1], padding='SAME')
    # conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')
    if bias:
        biases = tf.get_variable('bias', [num_outputs], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

    return conv