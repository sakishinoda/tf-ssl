"""
Implement a CNN Ladder.
Code starting point takerum vat-tf
Dropout on the pooling layers is replaced with batch norm.
Batch norm on convolution layers are carried out per-channel.
"""
import tensorflow as tf

def lrelu(x, a=0.1):
    if a < 1e-16:
        return tf.nn.relu(x)
    else:
        return tf.maximum(x, a * x)

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





from src.val import Encoder, get_batch_ops

class ConvolutionalEncoder(Encoder):
    def __init__(self, inputs, bn, is_training, params, this_encoder_noise=0.0,
            start_layer=0, update_batch_stats=True, scope='enc', reuse=None):

        self.layer_spec = self.make_layer_spec(params)
        self.lrelu_a = params.lrelu_a
        self.top_bn = params.top_bn

        super(ConvolutionalEncoder, self).__init__(
            inputs, bn, is_training, params,
            this_encoder_noise=this_encoder_noise,
            start_layer=start_layer, update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse)


    @staticmethod
    def make_layer_spec(params):
        types = params.cnn_layer_types
        init_size = params.cnn_init_size
        fan = params.cnn_fan
        ksizes = params.cnn_ksizes
        strides = params.cnn_strides

        dims = [init_size, ] * 4 + [init_size // 2, ] * 4 + [init_size // 4, ] * 4 + \
               [1, ]
        init_dim = fan[0]
        n_classes = fan[-1]

        layers = {}
        for l, type_ in enumerate(types):
            layers[l] = {'type': type_,
                         'dim': dims[l],
                         'ksize': ksizes[l],
                         'stride': strides[l],
                         'f_in': fan[l],
                         'f_out': fan[l + 1]
                         }

        return layers

    def create_layers(self, inputs, bn, is_training, start_layer,
                      update_batch_stats, scope, reuse):

        join, split_lu, labeled, unlabeled = get_batch_ops(self.batch_size)

        h = inputs + tf.random_normal(tf.shape(inputs)) * self.noise_sd

        layer_spec = self.layer_spec

        def split_moments(z_pre):
            z_pre_l, z_pre_u = split_lu(z_pre)
            # bn_axes = [0, 1, 2] if params.cnn else [0]
            bn_axes = list(range(len(z_pre.get_shape().as_list())-1))
            m_u, v_u = tf.nn.moments(z_pre_u, axes=bn_axes)
            return m_u, v_u, z_pre_l, z_pre_u

        def split_bn(z_pre, is_training, noise_sd=0.0):
            m_u, v_u, z_pre_l, z_pre_u = split_moments(z_pre)
            # if is_training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately

                if not update_batch_stats:
                    assert self.noise_sd > 0
                    # Corrupted encoder
                    # batch normalization + noise
                    bn_l = bn.batch_normalization(z_pre_l)
                    bn_u = bn.batch_normalization(z_pre_u, m_u, v_u)
                    z = join(bn_l, bn_u)
                    z += tf.random_normal(tf.shape(z_pre)) * self.noise_sd
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    bn_l = bn.update_batch_normalization(z_pre_l, l) if \
                        update_batch_stats else bn.batch_normalization(z_pre_l)
                    bn_u = bn.batch_normalization(z_pre_u, m_u, v_u)
                    z = join(bn_l, bn_u)
                return z

            def eval_batch_norm():
                mean = bn.ewma.average(bn.running_mean[l-1])
                var = bn.ewma.average(bn.running_var[l-1])
                z = bn.batch_normalization(z_pre, mean, var)
                return z

            z = tf.cond(is_training, training_batch_norm, eval_batch_norm)

            return z, m_u, v_u


        for l in range(1, self.num_layers+1):

            print("Layer {}: {} -> {}".format(
                l, layer_spec[l-1]['f_in'], layer_spec[l-1]['f_out']))

            self.labeled['h'][l-1], self.unlabeled['h'][l-1] = split_lu(h)

            if layer_spec[l-1]['type'] == 'c':
                h = conv(h,
                         ksize=layer_spec[l-1]['ksize'],
                         stride=1,
                         f_in=layer_spec[l-1]['f_in'],
                         f_out=layer_spec[l-1]['f_out'],
                         seed=None, name='c' + str(l-1))
                h, m, v = split_bn(h, is_training=is_training, noise_sd=self.noise_sd)
                h = lrelu(h, self.lrelu_a)

            elif layer_spec[l-1]['type'] == 'max':
                h = max_pool(h,
                             ksize=layer_spec[l-1]['ksize'],
                             stride=layer_spec[l-1]['stride'])
                h, m, v = split_bn(h, is_training=is_training, noise_sd=self.noise_sd)

            elif layer_spec[l-1]['type'] == 'avg':
                # Global average poolingg
                h = tf.reduce_mean(h, reduction_indices=[1, 2])
                m, v, _, _ = split_moments(h)

            elif layer_spec[l-1]['type'] == 'fc':
                h = fc(h, layer_spec[l-1]['f_in'],
                       layer_spec[l-1]['f_out'],
                       seed=None,
                       name='fc')
                if self.top_bn:
                    h, m, v = split_bn(h, is_training=is_training,
                                     noise_sd=self.noise_sd)
                else:
                    m, v, _, _ = split_moments(h)
            else:
                print('Layer type not defined')
                m, v, _, _ = split_moments(h)

            print(l, h.get_shape())
            self.labeled.z[l], self.unlabeled.z[l] = split_lu(h)
            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l], self.unlabeled.v[l] = m, v

        self.labeled.h[l], self.unlabeled.h[l] = split_lu(h)
        self.logits = h



def cnn_decoder(clean, corr, logits_corr, bn, combinator,
                 is_training,
                 params,
                 update_batch_stats=False,
                 layers=None):
        """
        Starts from logit (x has dim 10)

        """

        # init_size = 28  # MNIST
        # init_size = 32  # CIFAR-10

        z_est = {}
        d_cost = []

        if layers is None:
            layers = make_layer_spec(params)

        def deconv_bn_lrelu(h, l):
            """Inherits layers, PARAMS, is_training, update_batch_stats from outer environment."""
            h = deconv(h, ksize=layers[l]['ksize'], stride=layers[l]['stride'],
                       f_in=layers[l]["f_out"], f_out=layers[l]["f_in"],
                       name=layers[l]['type'] + str(l))

            h = bn(h, layers[l]["f_in"], is_training=is_training,
                   update_batch_stats=update_batch_stats, name='b' + str(l))

            h = lrelu(h, params.lrelu_a)
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


        # 10: fc
        # Dense batch x 10 -> batch x 192
        for l in range(params.num_layers, -1, -1):

            print("Layer {}: {} -> {}, denoising cost: {}".format(
                l, LS[l+1] if l+1<len(LS) else None,
                LS[l], denoising_cost[l]
            ))

            z, z_c = clean.unlabeled.z[l], corr.unlabeled.z[l]
            m, v = clean.unlabeled.m.get(l, 0), \
                   clean.unlabeled.v.get(l, 1-1e-10)

            type_ = layers[l-1]['type']
            print(type_)
            if l == params.num_layers:
                h = unlabeled(logits_corr)
            elif type_ == 'fc':
                h = fc(h,
                       dim_in=layers[l-1]["f_out"],
                       dim_out=layers[l-1]["f_in"],
                       seed=None, name=layers[l-1]['type'] + str(l))
            elif type_ == 'avg':
                # De-global mean pool with copying:
                # batch x 1 x 1 x 192 -> batch x 8 x 8 x 192
                h = tf.reshape(h, [-1, 1, 1, layers[l-1]["f_out"]])
                h = tf.tile(h, [1, layers[l-1]['dim'], layers[l-1]['dim'], 1])
            elif type_ == 'c':
                # Deconv
                h = deconv_bn_lrelu(h, l)
            elif type_ == 'max':
                # De-max-pool
                h = depool(h)

            else:
                print('Layer type not defined')

            h = bn.batch_normalization(h)
            z_est[l] = combinator(z_c, h, LS[l])
            z_est_bn = (z_est[l] - m) / v

            d_cost.append((tf.reduce_mean(tf.reduce_sum(
                tf.square(z_est_bn - z), 1)) / LS[l]) * denoising_cost[l])

        return z_est, d_cost


