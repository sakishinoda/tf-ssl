"""
Implement a CNN Ladder.
Code starting point takerum vat-tf
Dropout on the pooling layers is replaced with batch norm.
Batch norm on convolution layers are carried out per-channel.
"""
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--keep_prob_hidden', default=0.5, type=float)
parser.add_argument('--lrelu_a', default=0.1, type=float)
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--bn_stats_decay_factor', default=0.99, type=float)
parser.add_argument('--batch_size', default=100, type=int)
PARAMS = parser.parse_args()

def make_layer_spec(
    types = ('c', 'c', 'c', 'max', 'c', 'c', 'c', 'max', 'c', 'c', 'c', 'avg', 'fc'),
    fan = (3, 96, 96, 96, 96, 192, 192, 192, 192, 192, 192, 192, 192, 10),
    ksizes = (3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, None, None),
    strides = (1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, None, None),
    init_size = 32
):
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


def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, name="bn"):
    params_shape = (dim,)
    n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
    mean = tf.reduce_mean(x, axis)
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


def encoder(x, is_training=True, update_batch_stats=True, stochastic=True,
       seed=1234, layers=None):
    """
    Returns logit (pre-softmax)

    VAT Conv-Large:
    layer_sizes = [
    128, 128, 128, 128,
    256, 256, 256, 256,
    512, 256, 128]
    kernel_sizes = [
    3, 3, 3, 2,
    3, 3, 3, 2,
    3, 1, 1
    ]

    Ladder Conv-Large (similar to VAT Conv-Small on CIFAR-10)
    layer_sizes = [
    96, 96, 96, 96,
    192, 192, 192, 192,
    192, 192, 10]
    kernel_sizes = [
    3, 3, 3, 2,
    3, 3, 3, 2,
    3, 1, 1
    ]
    """

    h = x
    if layers is None:
        layers = make_layer_spec()
    # rng = np.random.RandomState(seed)

    def conv_bn_lrelu(h, l, f_in, f_out, ksize=3):
        h = conv(h, ksize=ksize, stride=1, f_in=f_in, f_out=f_out, seed=None,
             name='c'+str(l))
        h = lrelu(bn(h, f_out, is_training=is_training,
                     update_batch_stats=update_batch_stats, name='b'+str(l)),
                  PARAMS.lrelu_a)
        return h
    with tf.variable_scope('enc'):
        for l in range(len(layers.keys())):
            if layers[l]['type'] == 'c':
                h = conv_bn_lrelu(h, l, layers[l]['f_in'], layers[l]['f_out'],
                                  ksize=layers[l]['ksize'])
            elif layers[l]['type'] == 'max':
                h = max_pool(h,
                             ksize=layers[l]['ksize'],
                             stride=layers[l]['stride'])
                h = bn(h, dim=layers[l]['f_in'], is_training=is_training,
                       update_batch_stats=update_batch_stats, name='b' + str(l))
                # h = tf.nn.dropout(h, keep_prob=PARAMS.keep_prob_hidden, seed=None) if stochastic else h
            elif layers[l]['type'] == 'avg':
                h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average poolig
            elif layers[l]['type'] == 'fc':
                h = fc(h, layers[l]['f_in'], layers[l]['f_out'], seed=None,
                       name='fc')
                if PARAMS.top_bn:
                    h = bn(h, 10, is_training=is_training,
                           update_batch_stats=update_batch_stats, name='b'+str(l))
            else:
                print('Layer type not defined')

            print(l, h.get_shape())

    return h


def decoder(x, is_training=True, update_batch_stats=False, stochastic=True,
       seed=1234, batch_size=100, layers=None):
    """
    Starts from logit (x has dim 10)

    """

    # init_size = 28  # MNIST
    # init_size = 32  # CIFAR-10
    if layers is None:
        layers = make_layer_spec()

    h = x  # batch x 1 x 1 x 10

    def deconv_bn_lrelu(h, l):
        """Inherits layers, PARAMS, is_training, update_batch_stats from outer environment."""
        h = deconv(h, ksize=layers[l]['ksize'], stride=layers[l]['stride'],
                   f_in=layers[l]["f_out"], f_out=layers[l]["f_in"],
                   name=layers[l]['type'] + str(l))

        h = bn(h, layers[l]["f_in"], is_training=is_training,
               update_batch_stats=update_batch_stats, name='b' + str(l))

        h = lrelu(h, PARAMS.lrelu_a)
        return h

    def depool(h):
        """Deconvolution with a filter of ones and stride 2 upsamples with
        copying to double the size."""
        output_shape = h.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        c = output_shape[-1]
        h = tf.nn.conv2d_transpose(
            h,
            filter=tf.ones([2, 2, c, c]),
            output_shape=output_shape,
            padding='SAME',
            strides=[1, 2, 2, 1],
            data_format='NHWC'
        )
        return h

    with tf.variable_scope('dec'):
        # 10: fc
        # Dense batch x 10 -> batch x 192
        for l in reversed(range(len(layers.keys()))):
            type_ = layers[l]['type']
            if type_ == 'fc':
                h = fc(h, dim_in=layers[l]["f_out"], dim_out=layers[l]["f_in"],
                       seed=None, name=layers[l]['type'] + str(l))
            elif type_ == 'avg':
                # De-global mean pool with copying:
                # batch x 1 x 1 x 192 -> batch x 8 x 8 x 192
                h = tf.reshape(h, [-1, 1, 1, layers[l]["f_out"]])
                h = tf.tile(h, [1, layers[l]['dim'], layers[l]['dim'], 1])
            elif type_ == 'c':
                # Deconv
                h = deconv_bn_lrelu(h, l)
            elif type_ == 'max':
                # De-max-pool
                h = depool(h)
                h = bn(h, dim=layers[l]["f_out"], is_training=is_training,
                       update_batch_stats=update_batch_stats,
                       name=layers[l]['type'] + str(l))
            else:
                print('Layer type not defined')

            print(l, h.get_shape())

    return h



