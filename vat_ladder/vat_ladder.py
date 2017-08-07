# -----------------------------
# IMPORTS
# -----------------------------
# import IPython
import tensorflow as tf
import input_data
import os
# from tqdm import tqdm
import numpy as np

import time
from src import *
from conv_ladder import *

# -----------------------------
# ENCODERS
# -----------------------------

def vat_corrupter(inputs, noise_std, bn, is_training,
                update_batch_stats=True, layers=None, start_layer=1,
                  clean_logits=None):
    """
    is_training has to be a placeholder TF boolean
    Note: if is_training is false, update_batch_stats is false, since the
    update is only called in the training setting
    """
    r_vadv = generate_virtual_adversarial_perturbation(
        inputs, clean_logits, is_training=is_training, start_layer=0)

    h = inputs + r_vadv  # add noise to input
    d = {}  # to store the pre-activation, activation, mean and variance for each layer
    # The data for labeled and unlabeled examples are stored separately
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)

    for l in range(start_layer, params.num_layers+1):
        print("Layer ", l, ": ", LS[l - 1], " -> ", LS[l])
        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
        z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
        z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

        m, v = tf.nn.moments(z_pre_u, axes=[0])

        # if training:
        def training_batch_norm():
            # Training batch normalization
            # batch normalization for labeled and unlabeled examples is performed separately
            # if noise_std > 0:
            if not update_batch_stats:
                # Corrupted encoder
                # batch normalization + noise
                z = join(bn.batch_normalization(z_pre_l), bn.batch_normalization(z_pre_u, m, v))
                z += generate_virtual_adversarial_perturbation(
                    z_pre, clean_logits, is_training=is_training, start_layer=l)
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                bn_l = bn.update_batch_normalization(z_pre_l, l) if \
                    update_batch_stats else bn.batch_normalization(z_pre_l)
                bn_u = bn.batch_normalization(z_pre_u, m, v)
                z = join(bn_l, bn_u)
            return z

        # else:
        def eval_batch_norm():
            # Evaluation batch normalization
            # obtain average mean and variance and use it to normalize the batch
            mean = bn.ewma.average(bn.running_mean[l-1])
            var = bn.ewma.average(bn.running_var[l-1])
            z = bn.batch_normalization(z_pre, mean, var)
            # Instead of the above statement, the use of the following 2 statements containing a typo
            # consistently produces a 0.2% higher accuracy for unclear reasons.
            # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
            # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
            return z

        # perform batch normalization according to value of boolean "training" placeholder:
        z = tf.cond(is_training, training_batch_norm, eval_batch_norm)
        # z = training_batch_norm() if is_training else eval_batch_norm()

        if l == params.num_layers:
            # use softmax activation in output layer
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
        else:
            # use ReLU activation in hidden layers
            h = tf.nn.relu(z + weights["beta"][l-1])

        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
        d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)

    return h, d


def mlp_encoder(inputs, noise_std, bn, is_training,
                update_batch_stats=True, layers=None, start_layer=1,
                corrupt=None,
                clean_logits=None):
    """
    is_training has to be a placeholder TF boolean
    Note: if is_training is false, update_batch_stats is false, since the
    update is only called in the training setting
    """
    def generate_noise(start_layer):
        if corrupt == 'vatgauss':
            noise = generate_virtual_adversarial_perturbation(
                inputs, clean_logits, is_training=is_training,
                start_layer=start_layer) + \
                tf.random_normal(tf.shape(inputs)) * noise_std
        elif corrupt == 'vat':
            noise = generate_virtual_adversarial_perturbation(
                inputs, clean_logits, is_training=is_training,
                start_layer=start_layer)
        elif corrupt == 'gauss':
            noise = tf.random_normal(tf.shape(inputs)) * noise_std
        else:
            noise = tf.zeros(tf.shape(inputs))
        return noise

    h = inputs + generate_noise(1) # add noise to input
    d = {}  # to store the pre-activation, activation, mean and variance for each layer
    # The data for labeled and unlabeled examples are stored separately
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)

    for l in range(start_layer, params.num_layers+1):
        print("Layer ", l, ": ", LS[l - 1], " -> ", LS[l])
        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
        z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
        z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

        m, v = tf.nn.moments(z_pre_u, axes=[0])

        # if training:
        def training_batch_norm():
            # Training batch normalization
            # batch normalization for labeled and unlabeled examples is performed separately
            # if noise_std > 0:
            if not update_batch_stats:
                # Corrupted encoder
                # batch normalization + noise
                z = join(bn.batch_normalization(z_pre_l), bn.batch_normalization(z_pre_u, m, v))
                noise = generate_noise(l+1)
                z += noise
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                bn_l = bn.update_batch_normalization(z_pre_l, l) if \
                    update_batch_stats else bn.batch_normalization(z_pre_l)
                bn_u = bn.batch_normalization(z_pre_u, m, v)
                z = join(bn_l, bn_u)
            return z

        # else:
        def eval_batch_norm():
            # Evaluation batch normalization
            # obtain average mean and variance and use it to normalize the batch
            mean = bn.ewma.average(bn.running_mean[l-1])
            var = bn.ewma.average(bn.running_var[l-1])
            z = bn.batch_normalization(z_pre, mean, var)
            # Instead of the above statement, the use of the following 2 statements containing a typo
            # consistently produces a 0.2% higher accuracy for unclear reasons.
            # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
            # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
            return z

        # perform batch normalization according to value of boolean "training" placeholder:
        z = tf.cond(is_training, training_batch_norm, eval_batch_norm)
        # z = training_batch_norm() if is_training else eval_batch_norm()

        if l == params.num_layers:
            # use softmax activation in output layer
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
        else:
            # use ReLU activation in hidden layers
            h = tf.nn.relu(z + weights["beta"][l-1])

        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
        d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)

    return h, d


def cnn_encoder(x, noise_std, bn, is_training,
                update_batch_stats=True, layers=None):

    h = x + tf.random_normal(tf.shape(x)) * noise_std
    d = {
        'labeled': {
            'z': {}, # pre-activation
            'm': {}, # mean
            'v': {}, # variance
            'h': {}  # activation
        },
        'unlabeled': {
            'z': {},  # pre-activation
            'm': {},  # mean
            'v': {},  # variance
            'h': {}  # activation
        }
    }

    if layers is None:
        layers = make_layer_spec(params)

    def split_moments(z_pre):
        z_pre_l, z_pre_u = split_lu(z_pre)
        # bn_axes = [0, 1, 2] if params.cnn else [0]
        bn_axes = list(range(len(z_pre.get_shape().as_list())-1))
        m_u, v_u = tf.nn.moments(z_pre_u, axes=bn_axes)
        return m_u, v_u, z_pre_l, z_pre_u

    def split_bn(z_pre, is_training, noise_std=0.0):
        m_u, v_u, z_pre_l, z_pre_u = split_moments(z_pre)
        # if is_training:
        def training_batch_norm():
            # Training batch normalization
            # batch normalization for labeled and unlabeled examples is performed separately
            # if noise_std > 0:
            if not update_batch_stats:
                assert noise_std > 0
                # Corrupted encoder
                # batch normalization + noise
                bn_l = bn.batch_normalization(z_pre_l)
                bn_u = bn.batch_normalization(z_pre_u, m_u, v_u)
                z = join(bn_l, bn_u)
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
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


    for l in range(1, params.num_layers+1):

        print("Layer {}: {} -> {}".format(
            l, layers[l-1]['f_in'], layers[l-1]['f_out']))

        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)

        if layers[l-1]['type'] == 'c':
            h = conv(h,
                     ksize=layers[l-1]['ksize'],
                     stride=1,
                     f_in=layers[l-1]['f_in'],
                     f_out=layers[l-1]['f_out'],
                     seed=None, name='c' + str(l-1))
            h, m, v = split_bn(h, is_training=is_training, noise_std=noise_std)
            h = lrelu(h, params.lrelu_a)

        elif layers[l-1]['type'] == 'max':
            h = max_pool(h,
                         ksize=layers[l-1]['ksize'],
                         stride=layers[l-1]['stride'])
            h, m, v = split_bn(h, is_training=is_training, noise_std=noise_std)

        elif layers[l-1]['type'] == 'avg':
            # Global average poolingg
            h = tf.reduce_mean(h, reduction_indices=[1, 2])
            m, v, _, _ = split_moments(h)

        elif layers[l-1]['type'] == 'fc':
            h = fc(h, layers[l-1]['f_in'],
                   layers[l-1]['f_out'],
                   seed=None,
                   name='fc')
            if params.top_bn:
                h, m, v = split_bn(h, is_training=is_training,
                                 noise_std=noise_std)
            else:
                m, v, _, _ = split_moments(h)
        else:
            print('Layer type not defined')
            m, v, _, _ = split_moments(h)

        print(l, h.get_shape())
        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(h)
        # save mean and variance of unlabeled examples for decoding
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v

    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)

    return h, d


# -----------------------------
# DECODERS
# -----------------------------

def mlp_decoder(clean, corr, logits_corr, bn, combinator,
                is_training,
                params,
                update_batch_stats=False,
                layers=None
                ):
    z_est = {}
    d_cost = []  # to store the denoising cost of all layers
    for l in range(params.num_layers, -1, -1):
        print("Layer ", l, ": ", LS[l + 1] if l + 1 < len(LS) else
        None, " -> ", LS[l], ", denoising cost: ", denoising_cost[l])

        z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
        m, v = clean['unlabeled']['m'].get(l, 0), \
               clean['unlabeled']['v'].get(l, 1 - 1e-10)
        # print(l)
        if l == params.num_layers:
            u = unlabeled(logits_corr)
        else:
            u = tf.matmul(z_est[l+1], weights['V'][l])

        u = bn.batch_normalization(u)

        z_est[l] = combinator(z_c, u, LS[l])

        z_est_bn = (z_est[l] - m) / v
        # append the cost of this layer to d_cost
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / LS[l]) * denoising_cost[l])

    return z_est, d_cost


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

            z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
            m, v = clean['unlabeled']['m'].get(l, 0), \
                   clean['unlabeled']['v'].get(l, 1-1e-10)

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



# -----------------------------
# BATCH NORMALIZATION SETUP
# -----------------------------
class BatchNormLayers(object):
    def __init__(self, ls, params, scope='bn'):
        self.params = params
        self.bn_assigns = []  # this list stores the updates to be made to average mean and variance
        self.ewma = tf.train.ExponentialMovingAverage(
            decay=0.99)  # to calculate the moving averages of mean and variance

        # average mean and variance of all layers
        with tf.variable_scope(scope):
            self.running_var = [tf.get_variable(
                'v'+str(i),
                initializer=tf.constant(1.0, shape=[l]),
                trainable=False) for i,l in enumerate(ls[1:])]
            self.running_mean = [tf.get_variable(
                'm'+str(i),
                initializer=tf.constant(0.0, shape=[l]),
                trainable=False) for i,l in enumerate(ls[1:])]

    def update_batch_normalization(self, batch, l):
        """
        batch normalize + update average mean and variance of layer l
        if CNN, use channel-wise batch norm
        """

        # bn_axes = [0, 1, 2] if self.params.cnn else [0]
        bn_axes = list(range(len(batch.get_shape().as_list())-1))
        mean, var = tf.nn.moments(batch, axes=bn_axes)
        print(l, mean.get_shape().as_list(),
              self.running_mean[l-1].get_shape().as_list(),
              batch.get_shape().as_list())

        assign_mean = self.running_mean[l-1].assign(mean)
        assign_var = self.running_var[l-1].assign(var)
        self.bn_assigns.append(
            self.ewma.apply([self.running_mean[l-1], self.running_var[l-1]]))

        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)


    def batch_normalization(self, batch, mean=None, var=None):
        # bn_axes = [0, 1, 2] if self.params.cnn else [0]
        bn_axes = list(range(len(batch.get_shape().as_list())-1))
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=bn_axes)

        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

# -----------------------------
# RECOMBINATION FUNCTIONS
# -----------------------------

def amlp_combinator(z_c, u, size):
    uz = tf.multiply(z_c, u)
    x = tf.stack([z_c, u, uz], axis=-1)
    print(size)
    # print(z_c.get_shape, u.get_shape, uz.get_shape)

    h = fclayer(x, size_out=4, wts_init=tf.random_normal_initializer(
        stddev=params.combinator_sd), reuse=None) #, scope='combinator_hidden')

    o = fclayer(h, size_out=1, wts_init=tf.random_normal_initializer(
        stddev=params.combinator_sd), reuse=None,
                activation=tf.nn.relu) #, scope='combinator_out')

    return tf.squeeze(o)


def gauss_combinator(z_c, u, size):
    "gaussian denoising function proposed in the original paper"
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est


# -----------------------------
# VAT FUNCTIONS
# -----------------------------

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp

def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))

def forward(x, is_training, update_batch_stats=False, seed=1234, start_layer=1):

    def training_logit():
        print("=== VAT Clean Pass === ")
        logit, _ = encoder(x, 0.0, bn,
                          is_training=is_training,
                          update_batch_stats=update_batch_stats,
                          start_layer=start_layer)
        return logit

    def testing_logit():
        print("=== VAT Corrupted Pass ===")
        logit, _ = encoder(x, params.encoder_noise_sd, bn,
                           is_training=is_training,
                           update_batch_stats=update_batch_stats,
                           start_layer=start_layer)
        return logit

    # return tf.cond(is_training, training_logit, testing_logit)
    return training_logit()

def get_normalized_vector(d):

    d_dims = len(d.get_shape()) - 1
    axes = [range(1, d_dims)] if d_dims > 1 else [1]
    d /= (1e-12 + tf.reduce_max(tf.abs(d), axis=axes, keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), axis=axes,
                                      keep_dims=True))
    return d

def generate_virtual_adversarial_perturbation(x, logit, is_training,
                                              start_layer=1):
    d = tf.random_normal(shape=tf.shape(x))
    for k in range(params.num_power_iterations):
        d = params.xi * get_normalized_vector(d)
        logit_p = logit
        print("=== Power Iteration: {} ===".format(k))
        logit_m = forward(x + d, update_batch_stats=False,
                          is_training=is_training, start_layer=start_layer)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)
    return params.epsilon * get_normalized_vector(d)

def virtual_adversarial_loss(x, logit, is_training, name="vat_loss",
                             start_layer=1):
    print("=== VAT Pass: Generating VAT perturbation ===")
    r_vadv = generate_virtual_adversarial_perturbation(
        x, logit, is_training=is_training, start_layer=start_layer)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    print("=== VAT Pass: Computing VAT Loss (KL Divergence)")
    logit_m = forward(x + r_vadv, update_batch_stats=False,
                      is_training=is_training, start_layer=start_layer)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)


# def main():
# -----------------------------
# PARAMETER PARSING
# -----------------------------

params = process_cli_params(get_cli_params())

# Set GPU device to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(params.which_gpu)

# Set seeds
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

print("===  Loading Data ===")
mnist = input_data.read_data_sets("MNIST_data", n_labeled=params.num_labeled, one_hot=True)
num_examples = mnist.train.num_examples

starter_learning_rate = params.initial_learning_rate

# epoch after which to begin learning rate decay
decay_after = params.decay_start_epoch
batch_size = params.batch_size
num_iter = (num_examples//batch_size) * params.end_epoch  # number of loop iterations

# -----------------------------
# LADDER SETUP
# -----------------------------

# Set layer sizes for encoders
# Choose encoder/decoder
# Make appropriately sized placeholder
if params.cnn:
    encoder, decoder = cnn_encoder,  cnn_decoder
    layers = make_layer_spec(params)
    LS = params.cnn_fan
else:
    encoder = mlp_encoder
    decoder = mlp_decoder
    layers = None
    LS = params.encoder_layers
    shapes = list(zip(LS[:-1], LS[1:]))  # shapes of linear layers

inputs_ph = tf.placeholder(tf.float32, shape=(None, LS[0]))
outputs = tf.placeholder(tf.float32)

inputs = tf.reshape(inputs_ph, shape=[
        -1, params.cnn_init_size, params.cnn_init_size, params.cnn_fan[0]
    ]) if params.cnn else inputs_ph

weights = {# batch normalization parameter to shift the normalized value
           'beta': [bias_init(0.0, LS[l + 1], "beta")
                    for l in range(params.num_layers)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bias_init(1.0, LS[l + 1], "beta")
                     for l in range(params.num_layers)]}

if not params.cnn:
    weights.update({
        'W': [wts_init(s, "W") for s in shapes],  # Encoder weights
        'V': [wts_init(s[::-1], "V") for s in shapes],  # Decoder weights
        })

# scaling factor for noise used in corrupted encoder
noise_std = params.encoder_noise_sd

# hyperparameters that denote the importance of each layer
denoising_cost = params.rc_weights if not params.cnn else ([1.,] * len(LS))

# Lambdas for extracting labeled/unlabeled, etc.
join = lambda l, u: tf.concat([l, u], 0)
split_lu = lambda x: (labeled(x), unlabeled(x))
labeled = lambda x: x[:batch_size] if x is not None else x
unlabeled = lambda x: x[batch_size:] if x is not None else x
# labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
# unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x

# Boolean training flag
train_flag = tf.placeholder(tf.bool)

# Set up Batch Norm
bn = BatchNormLayers(LS, params)

print( "=== Clean Encoder ===")
with tf.variable_scope('enc', reuse=None):
    logits_clean, clean = encoder(inputs, 0.0, bn, is_training=train_flag,
                                  update_batch_stats=True, layers=layers,
                                  corrupt=None)

print( "=== Corrupted Encoder === ")
with tf.variable_scope('enc', reuse=True):
    logits_corr, corr = encoder(inputs, noise_std, bn, is_training=train_flag,
                                update_batch_stats=False, layers=layers,
                                corrupt=params.corrupt,
                                clean_logits=logits_clean)
#  add noise

# -----------------------------
# DECODER
# -----------------------------
# Choose recombination function
combinator = gauss_combinator


print( "=== Decoder ===")
with tf.variable_scope('dec', reuse=None):
    z_est, d_cost = decoder(clean, corr, logits_corr, bn, combinator,
                            is_training=train_flag, params=params)

# -----------------------------
# PUTTING IT ALL TOGETHER
# -----------------------------
# vat cost
# ul_x = unlabeled(inputs)
# ul_logit = unlabeled(logits_corr)
# ul_logit = forward(ul_x, is_training=True, update_batch_stats=False)

vat_loss = params.vat_weight * virtual_adversarial_loss(
            inputs, logits_corr, is_training=train_flag)

if params.vat_rc:
    for l in range(1, params.num_layers):
        l_inputs = join(corr['labeled']['h'][l-1], corr['unlabeled']['h'][l-1])
        vat_loss += (params.vat_weight *
                     denoising_cost[l] *
                     virtual_adversarial_loss(
                         l_inputs, logits_corr,
                         is_training=train_flag, start_layer=l))


ent_loss = params.ent_weight * entropy_y_x(logits_corr)

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(logits_corr)
cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost

loss = cost + u_cost + vat_loss + ent_loss # total cost

pred_cost = -tf.reduce_mean(
    tf.reduce_sum(outputs * tf.log(logits_clean), 1))  # cost used for prediction

correct_prediction = tf.equal(
    tf.argmax(logits_clean, 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn.bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=5)

# -----------------------------
print("===  Starting Session ===")
sess = tf.Session()

i_iter = 0

# -----------------------------
# Resume from checkpoint
ckpt_dir = "checkpoints/" + params.id + "/"
ckpt = tf.train.get_checkpoint_state(ckpt_dir)  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
    i_iter = (epoch_n+1) * (num_examples//batch_size)
    print("Restored Epoch ", epoch_n)
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    init = tf.global_variables_initializer()
    sess.run(init)

# -----------------------------
# Write logs to appropriate directory
log_dir = "logs/" + params.id
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

desc_file = log_dir + "/" + "description"
with open(desc_file, 'a') as f:
    print(*order_param_settings(params), sep='\n', file=f, flush=True)
    print("Trainable parameters:", count_trainable_params(), file=f,
          flush=True)

log_file = log_dir + "/" + "train_log"


# -----------------------------
print("=== Training ===")

[init_acc, init_loss] = sess.run([accuracy, loss], feed_dict={
    inputs_ph: mnist.train.labeled_ds.images, outputs:
        mnist.train.labeled_ds.labels,
    train_flag: False})
print("Initial Train Accuracy: ", init_acc, "%")
print("Initial Train Loss: ", init_loss)

[init_acc] = sess.run([accuracy], feed_dict={
    inputs_ph: mnist.test.images, outputs: mnist.test.labels, train_flag:
        False})
print("Initial Test Accuracy: ", init_acc, "%")
# print("Initial Test Loss: ", init_loss)


start = time.time()
# for i in tqdm(range(i_iter, num_iter)):
for i in range(i_iter, num_iter):
    images, labels = mnist.train.next_batch(batch_size)

    _ = sess.run(
        [train_step],
        feed_dict={inputs_ph: images, outputs: labels, train_flag: True})


    if (i > 1) and ((i+1) % (params.test_frequency_in_epochs*(
                num_iter//params.end_epoch)) == 0):
        now = time.time() - start
        epoch_n = i//(num_examples//batch_size)
        if (epoch_n+1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (params.end_epoch - (epoch_n + 1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0., ratio / (params.end_epoch - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, ckpt_dir + 'model.ckpt', epoch_n)
        # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"

        with open(log_file, 'a') as train_log:
            # write test accuracy to file "train_log"
            # train_log_w = csv.writer(train_log)
            log_i = [now, epoch_n] + sess.run(
                [accuracy],
                feed_dict={inputs_ph: mnist.test.images,
                           outputs: mnist.test.labels, train_flag: False}
            ) + sess.run(
                [loss, cost, u_cost, vat_loss, ent_loss],
                feed_dict={inputs_ph: images, outputs: labels, train_flag:
                    True})
            # train_log_w.writerow(log_i)
            print(*log_i, sep=',', flush=True, file=train_log)

print("Final Accuracy: ", sess.run(accuracy, feed_dict={
    inputs: mnist.test.images, outputs: mnist.test.labels, train_flag: False}),
      "%")

sess.close()


# if __name__ == '__main__':
#     main()