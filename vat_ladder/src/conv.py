"""
Implement a CNN Ladder.
Code starting point takerum vat-tf
Dropout on the pooling layers is replaced with batch norm.
Batch norm on convolution layers are carried out per-channel.
"""
import tensorflow as tf
import math

def lrelu(x, a=0.1):
    if a < 1e-16:
        return tf.nn.relu(x)
    else:
        return tf.maximum(x, a * x)

def fc(x, dim_in, dim_out, seed=None, name='fc', scope=None, reuse=None):

    num_units_in = dim_in
    num_units_out = dim_out
    # weights_initializer = tf.contrib.layers.variance_scaling_initializer(
    #     seed=seed)
    weights_initializer = tf.random_normal_initializer(stddev=1./math.sqrt(dim_in))

    if scope is None:
        scope = tf.get_variable_scope()

    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable(name + '_W',
                                shape=[num_units_in, num_units_out],
                                initializer=weights_initializer)
        biases = tf.get_variable(name + '_b',
                                 shape=[num_units_out],
                                 initializer=tf.constant_initializer(0.0))
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x


def conv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False,
         seed=None, name='conv', scope=None, reuse=None):
    shape = [ksize, ksize, f_in, f_out]
    # As used in VAT
    # initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    # As used in Ladder

    if padding == 'FULL' and ksize > 1:
        p = ksize-1
        x  = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        padding = 'VALID'


    bound = math.sqrt(3.0 / max(1.0, (ksize*ksize*f_in)))
    initializer = tf.random_uniform_initializer(minval=-bound, maxval=bound)

    if scope is None:
        scope = tf.get_variable_scope()

    with tf.variable_scope(scope, reuse=reuse):
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
           seed=None, name='deconv', scope=None, reuse=None):


    out_shape = x.get_shape().as_list()
    out_shape[-1] = f_out
    w_shape = [ksize, ksize, f_out, f_in]  # deconv requires f_out, f_in
    bound = math.sqrt(3.0 / max(1.0, (ksize*ksize*f_out)))
    initializer = tf.random_uniform_initializer(minval=-bound, maxval=bound)

    if scope is None:
        scope = tf.get_variable_scope()

    with tf.variable_scope(scope, reuse=reuse):

        weights = tf.get_variable(name + '_W',
                                  shape=w_shape,
                                  dtype='float',
                                  initializer=initializer)

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
                          padding='VALID')


def max_pool(x, ksize=2, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID')

