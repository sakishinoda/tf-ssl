"""
Implement a CNN Ladder.
Code starting point takerum vat-tf
Dropout on the pooling layers is replaced with batch norm.
Batch norm on convolution layers are carried out per-channel.
"""
import tensorflow as tf
# import numpy as np
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--keep_prob_hidden', default=0.5, type=float)
# parser.add_argument('--lrelu_a', default=0.1, type=float)
# parser.add_argument('--top_bn', action='store_true')
# parser.add_argument('--bn_stats_decay_factor', default=0.99, type=float)
# parser.add_argument('--batch_size', default=100, type=int)
# PARAMS = parser.parse_args()

def make_layer_spec(params):
    types = params.cnn_layer_types
    init_size = params.cnn_init_size
    fan = params.cnn_fan
    ksizes = params.cnn_ksizes
    strides = params.cnn_strides


    dims = [init_size, ] * 4 + [init_size//2, ] * 4 + [init_size//4, ] * 4 + \
           [1,]
    init_dim = fan[0]
    n_classes = fan[-1]

    layers = {}
    for l, type_ in enumerate(types):
        layers[l] = {'type': type_,
                     'dim': dims[l],
                     'ksize': ksizes[l],
                     'stride': strides[l],
                     'f_in': fan[l],
                     'f_out': fan[l+1]
                     }

    return layers

def lrelu(x, a=0.1):
    if a < 1e-16:
        return tf.nn.relu(x)
    else:
        return tf.maximum(x, a * x)


def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, name="bn", mean=None, var=None):
    params_shape = (dim,)
    n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
    if mean is None:
        mean = tf.reduce_mean(x, axis)
    if var is None:
        var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)

    avg_mean = tf.get_variable(
        name=name + "_mean",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
        trainable=False
    )

    avg_var = tf.get_variable(
        name=name + "_var",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections,
        trainable=False
    )

    gamma = tf.get_variable(
        name=name + "_gamma",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections
    )

    beta = tf.get_variable(
        name=name + "_beta",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
    )

    if is_training:
        avg_mean_assign_op = tf.no_op()
        avg_var_assign_op = tf.no_op()
        if update_batch_stats:
            avg_mean_assign_op = tf.assign(
                avg_mean,
                PARAMS.bn_stats_decay_factor * avg_mean + (1 - PARAMS.bn_stats_decay_factor) * mean)
            avg_var_assign_op = tf.assign(
                avg_var,
                PARAMS.bn_stats_decay_factor * avg_var + (n / (n - 1))
                * (1 - PARAMS.bn_stats_decay_factor) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            z = (x - mean) / tf.sqrt(1e-6 + var)
    else:
        z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

    return gamma * z + beta


def fc(x, dim_in, dim_out, seed=None, name='fc'):
    num_units_in = dim_in
    num_units_out = dim_out
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)

    weights = tf.get_variable(name + '_W',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer)
    biases = tf.get_variable(name + '_b',
                             shape=[num_units_out],
                             initializer=tf.constant_initializer(0.0))
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def conv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False, seed=None, name='conv'):
    shape = [ksize, ksize, f_in, f_out]
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    weights = tf.get_variable(name + '_W',
                            shape=shape,
                            dtype='float',
                            initializer=initializer)
    x = tf.nn.conv2d(x, filter=weights, strides=[1, stride, stride, 1],
                     padding=padding)

    if use_bias:
        bias = tf.get_variable(name + '_b',
                               shape=[f_out],
                               dtype='float',
                               initializer=tf.zeros_initializer)
        return tf.nn.bias_add(x, bias)
    else:
        return x


def deconv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False,
           seed=None, name='deconv'):

    w_shape = [ksize, ksize, f_out, f_in]  # deconv requires f_out, f_in
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    weights = tf.get_variable(name + '_W',
                              shape=w_shape,
                              dtype='float',
                              initializer=initializer)

    out_shape = x.get_shape().as_list()
    out_shape[-1] = f_out
    # print(weights.get_shape().as_list(), x.get_shape().as_list(), out_shape)

    x = tf.nn.conv2d_transpose(x, weights,
                               output_shape=out_shape,
                               strides=[1, stride, stride, 1],
                               padding=padding)

    if use_bias:
        bias = tf.get_variable(name + '_b',
                               shape=[f_out],
                               dtype='float',
                               initializer=tf.zeros_initializer)
        return tf.nn.bias_add(x, bias)
    else:
        return x

def avg_pool(x, ksize=2, stride=2):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def max_pool(x, ksize=2, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')





