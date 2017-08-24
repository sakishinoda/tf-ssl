"""Virtual Adversarial Ladder"""
import tensorflow as tf
from src.utils import count_trainable_params, preprocess, get_batch_ops
from src.conv import *
import tensorflow.contrib.layers as layers
import math


VERBOSE = False

# -----------------------------
# -----------------------------
# LADDER CLASSES
# -----------------------------
# -----------------------------
class Activations(object):
    """Store statistics for each layer in the encoder/decoder structures

    Attributes
    ----------
        z, dict: pre-activation, used for reconstruction
        h, dict: activations
        m, dict: mean of each layer activations
        v, dict: variance of each layer activations

    """

    def __init__(self):
        self.z = {}  # pre-activation
        self.h = {}  # activation
        self.m = {}  # mean
        self.v = {}  # variance


# -----------------------------
# ENCODER
# -----------------------------
class Encoder(object):
    """MLP Encoder

    Arguments
    ---------
        inputs: tensor
        bn: BatchNormLayers
        is_training: tensorflow bool
        params: argparse Namespace
            with attributes
            encoder_layers (sequence of ints), and
            batch_size (int)
        this_encoder_noise: float, default 0.0
        start_layer: int, default 0
        update_batch_stats: bool
        scope: str, default 'enc'
        reuse: bool or None, default None

    Attributes
    ----------
        bn: reference to batch norm class
        start_layer
        is_training: tf bool
        encoder_layer: list
        batch_size: int
        noise_sd: float
        logits: pre-softmax output at final layer
        labeled: an Activations object with attributes z, h, m, v
        unlabeled: an Activations object


    """

    def __init__(
            self, inputs, bn, is_training, params, this_encoder_noise=0.0,
            start_layer=0, update_batch_stats=True, scope='enc', reuse=None):

        self.inputs = inputs
        self.bn = bn
        self.start_layer = start_layer
        self.is_training = is_training

        self.lrelu_a = params.lrelu_a
        self.batch_size = params.batch_size
        self.encoder_layers = params.encoder_layers
        self.top_bn = params.top_bn

        self.noise_sd = this_encoder_noise
        self.start_layer = start_layer
        self.update_batch_stats = update_batch_stats
        self.scope = scope
        self.reuse = reuse

        self.labeled = Activations()
        self.unlabeled = Activations()

        # encoder_layers = encoder_layers  # seq of layer sizes, len num_layers
        self.num_layers = len(params.encoder_layers) - 1

        self.create_layers(inputs, bn, is_training, start_layer,
                           update_batch_stats, scope, reuse)

    def create_layers(self, inputs, bn, is_training, start_layer,
                      update_batch_stats, scope, reuse):
        # inputs = self.inputs
        # bn = self.bn
        # is_training = self.is_training
        # start_layer = self.start_layer
        # update_batch_stats = self.update_batch_stats
        # scope = self.scope
        # reuse = self.reuse

        el = self.encoder_layers

        join, split_lu, labeled, unlabeled = get_batch_ops(self.batch_size)
        # Layer 0: inputs, size 784
        h = inputs + self.generate_noise(inputs, start_layer)
        self.labeled.z[start_layer], self.unlabeled.z[start_layer] = split_lu(h)

        for l_out in range(start_layer + 1, self.num_layers + 1):
            l_in = l_out - 1
            # init_sd = 1 / math.sqrt(el[l_in]) # ladder
            init_sd = 1 / math.sqrt(el[l_in] + el[l_out])  # vat
            self.print_progress(l_out)

            self.labeled.h[l_in], self.unlabeled.z[l_in] = split_lu(h)
            # z_pre = tf.matmul(h, self.W[l-1])
            z_pre = layers.fully_connected(
                h,
                num_outputs=el[l_out],
                weights_initializer=tf.random_normal_initializer(
                    stddev=init_sd),
                biases_initializer=None,
                activation_fn=None,
                scope=scope + str(l_out),
                reuse=reuse)

            z_pre_l, z_pre_u = split_lu(z_pre)
            # bn_axes = list(range(len(z_pre_u.get_shape().as_list())))
            bn_axes = [0]
            m, v = tf.nn.moments(z_pre_u, axes=bn_axes)

            # if training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is
                # performed separately
                # if noise_sd > 0:
                if self.noise_sd > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = join(bn.batch_normalization(z_pre_l),
                             bn.batch_normalization(z_pre_u, m, v))
                    noise = self.generate_noise(z_pre, l_out)
                    z += noise
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance
                    # using batch mean and variance of labeled examples
                    bn_l = bn.update_batch_normalization(z_pre_l, l_out) if \
                        update_batch_stats else bn.batch_normalization(z_pre_l)
                    bn_u = bn.batch_normalization(z_pre_u, m, v)
                    z = join(bn_l, bn_u)
                return z

            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = bn.ema.average(bn.running_mean[l_in])
                var = bn.ema.average(bn.running_var[l_in])
                z = bn.batch_normalization(z_pre, mean, var)

                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(is_training, training_batch_norm, eval_batch_norm)

            if l_out == self.num_layers:
                # return pre-softmax logits in final layer
                self.logits = bn.gamma[l_in] * z + bn.beta[l_in]
                h = tf.nn.softmax(self.logits)

            elif self.lrelu_a > 0.0:
                h = lrelu(z + bn.beta[l_in])

            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + bn.beta[l_in])

            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l_out], self.unlabeled.v[l_out] = m, v
            self.labeled.z[l_out], self.unlabeled.z[l_out] = split_lu(z)
            self.labeled.h[l_out], self.unlabeled.h[l_out] = split_lu(h)

    def print_progress(self, l_out):
        el = self.encoder_layers
        if VERBOSE:
            print("Layer {}: {} -> {}".format(l_out, el[l_out - 1], el[l_out]))

    def generate_noise(self, inputs, l_out):
        return tf.random_normal(tf.shape(inputs)) * self.noise_sd


# VAT Encoder
class VATEncoder(Encoder):
    def create_layers(self, inputs, bn, is_training, start_layer,
                      update_batch_stats, scope, reuse):
        # inputs = self.inputs
        # bn = self.bn
        # is_training = self.is_training
        # start_layer = self.start_layer
        # update_batch_stats = self.update_batch_stats
        # scope = self.scope
        # reuse = self.reuse

        el = self.encoder_layers

        join, split_lu, labeled, unlabeled = get_batch_ops(self.batch_size)
        # Layer 0: inputs, size 784
        h = inputs + self.generate_noise(inputs, start_layer)
        self.labeled.z[start_layer], self.unlabeled.z[start_layer] = split_lu(h)

        for l_out in range(start_layer + 1, self.num_layers + 1):
            l_in = l_out - 1
            # init_sd = 1 / math.sqrt(el[l_in]) # ladder
            init_sd = 1 / math.sqrt(el[l_in] + el[l_out])  # vat
            self.print_progress(l_out)

            self.labeled.h[l_in], self.unlabeled.z[l_in] = split_lu(h)
            # z_pre = tf.matmul(h, self.W[l-1])
            z_pre = layers.fully_connected(
                h,
                num_outputs=el[l_out],
                weights_initializer=tf.random_normal_initializer(
                    stddev=init_sd),
                biases_initializer=None,
                activation_fn=None,
                scope=scope + str(l_out),
                reuse=reuse)

            m, v = tf.nn.moments(z_pre, axes=[0])

            # if training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is
                # performed separately

                z = bn.update_batch_normalization(z_pre, l_out) if \
                    update_batch_stats else bn.batch_normalization(z_pre)

                noise = self.generate_noise(z, l_out)
                z += noise

                return z

            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = bn.ema.average(bn.running_mean[l_in])
                var = bn.ema.average(bn.running_var[l_in])
                z = bn.batch_normalization(z_pre, mean, var)

                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(is_training, training_batch_norm, eval_batch_norm)

            if l_out == self.num_layers:
                # return pre-softmax logits in final layer
                if self.top_bn:
                    self.logits = bn.gamma[l_in] * z + bn.beta[l_in]
                else:
                    self.logits = z
                h = tf.nn.softmax(self.logits)

            elif self.lrelu_a > 0.0:
                h = lrelu(z + bn.beta[l_in])

            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + bn.beta[l_in])

            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l_out], self.unlabeled.v[l_out] = m, v
            self.labeled.z[l_out], self.unlabeled.z[l_out] = split_lu(z)
            self.labeled.h[l_out], self.unlabeled.h[l_out] = split_lu(h)



# VAN Encoder
class VirtualAdversarialNoiseEncoder(Encoder):
    def __init__(
            self, inputs, bn, is_training, params, clean_logits,
            this_encoder_noise=0.0, start_layer=0, update_batch_stats=True,
            scope='enc', reuse=None):

        self.params = params
        self.clean_logits = clean_logits

        super(VirtualAdversarialNoiseEncoder, self).__init__(
            inputs, bn, is_training, params,
            this_encoder_noise=this_encoder_noise,
            start_layer=start_layer,
            update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse
        )

    def get_vadv_noise(self, inputs, l):
        join, split_lu, labeled, unlabeled = get_batch_ops(self.batch_size)

        adv = Adversary(
            bn=self.bn,
            params=self.params,
            layer_eps=self.params.epsilon[l],
            start_layer=l
        )

        x = unlabeled(inputs)
        logit = unlabeled(self.clean_logits)

        ul_noise = adv.generate_virtual_adversarial_perturbation(
            x=x, logit=logit, is_training=self.is_training)

        return join(tf.zeros(tf.shape(labeled(inputs))), ul_noise)

    def print_progress(self, l_out):
        el = self.encoder_layers
        if VERBOSE:
            print("Layer {}: {} -> {}, epsilon {}".format(l_out, el[l_out - 1],
                                                      el[l_out],
                                                      self.params.epsilon.get(
                                                          l_out - 1)))

    def generate_noise(self, inputs, l):

        if self.noise_sd > 0.0:
            noise = tf.random_normal(tf.shape(inputs)) * self.noise_sd

            if self.params.model == "n" and l == 0:
                noise += self.get_vadv_noise(inputs, l)

            elif self.params.model == "nlw" and (l + 1 < self.num_layers):
                # don't add adversarial noise to logits
                noise += self.get_vadv_noise(inputs, l)

        else:
            noise = tf.zeros(tf.shape(inputs))

        return noise


# ConvEncoder

class ConvEncoder(Encoder):
    def __init__(self, inputs, bn, is_training, params, this_encoder_noise=0.0,
            start_layer=0, update_batch_stats=True, scope='enc', reuse=None):

        self.layer_spec = self.make_layer_spec(params)
        self.lrelu_a = params.lrelu_a
        self.top_bn = params.top_bn

        super(ConvEncoder, self).__init__(
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
        dims = params.cnn_dims
        # dims = [init_size, ] * 4 + [init_size // 2, ] * 4 + [init_size // 4, ] * 4 + \
        #        [1, ]
        # init_dim = fan[0]
        # n_classes = fan[-1]

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

        def split_bn(z_pre, is_training, l_out):
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
                    noise = self.generate_noise(z_pre, l_out)
                    z += noise
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    bn_l = bn.update_batch_normalization(z_pre_l, l_out) if \
                        update_batch_stats else bn.batch_normalization(z_pre_l)
                    bn_u = bn.batch_normalization(z_pre_u, m_u, v_u)
                    z = join(bn_l, bn_u)
                return z

            def eval_batch_norm():
                mean = bn.ema.average(bn.running_mean[l_out-1])
                var = bn.ema.average(bn.running_var[l_out-1])
                z = bn.batch_normalization(z_pre, mean, var)
                return z

            z = tf.cond(is_training, training_batch_norm, eval_batch_norm)

            return z, m_u, v_u


        for l_out in range(1, self.num_layers+1):
            l_in = l_out-1
            if VERBOSE:
                print("Layer {}: {} -> {}".format(
                l_out, layer_spec[l_in]['f_in'],
                layer_spec[l_out-1]['f_out']))

            self.labeled.h[l_in], self.unlabeled.h[l_in] = split_lu(h)

            # Convolutional layer
            if layer_spec[l_in]['type'] == 'c':
                z_pre = conv(h,
                         ksize=layer_spec[l_in]['ksize'],
                         stride=1,
                         f_in=layer_spec[l_in]['f_in'],
                         f_out=layer_spec[l_in]['f_out'],
                         seed=None, name='c' + str(l_in),
                         scope=self.scope, reuse=self.reuse)
                z, m, v = split_bn(
                    z_pre, is_training=is_training, l_out=l_out)

                if self.lrelu_a > 0.0:
                    h = lrelu(z + bn.beta[l_in], self.lrelu_a)
                else:
                    h = tf.nn.relu(z + bn.beta[l_in])

            # Max pooling layer
            elif layer_spec[l_in]['type'] == 'max':
                z_pre = max_pool(h,
                             ksize=layer_spec[l_in]['ksize'],
                             stride=layer_spec[l_in]['stride'])
                z, m, v = split_bn(
                    z_pre, is_training=is_training, l_out=l_out)
                h = z

            # Average pooling
            elif layer_spec[l_in]['type'] == 'avg':
                # Global average pooling
                z_pre = tf.reduce_mean(h, reduction_indices=[1, 2])
                m, v, _, _ = split_moments(z_pre)
                z = z_pre
                h = z


            # Fully connected layer
            elif layer_spec[l_in]['type'] == 'fc':
                z_pre = fc(h, layer_spec[l_in]['f_in'],
                       layer_spec[l_in]['f_out'],
                       seed=None, name='fc',
                       scope=self.scope, reuse=self.reuse)

                if self.top_bn:
                    z, m, v = split_bn(
                        z_pre, is_training=is_training, l_out=l_out)
                else:
                    m, v, _, _ = split_moments(z_pre)
                    z = z_pre

                if l_out == self.num_layers:
                    self.logits = bn.gamma[l_in] * z + bn.beta[l_in]
                    h = tf.nn.softmax(self.logits)
                elif self.lrelu_a > 0.0:
                    h = lrelu(z + bn.beta[l_in])
                else:
                    h = tf.nn.relu(z + bn.beta[l_in])


            else:
                if VERBOSE:
                    print('Layer type not defined')
                m, v, _, _ = split_moments(h)
                z = h


            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l_out], self.unlabeled.v[l_out] = m, v
            self.labeled.z[l_out], self.unlabeled.z[l_out] = split_lu(z)
            self.labeled.h[l_out], self.unlabeled.h[l_out] = split_lu(h)



# -----------------------------
# DECODER
# -----------------------------
class Decoder(object):
    """MLP Decoder

    Arguments
    ---------
        clean: Encoder object
        corr: Encoder object
        bn: BatchNormLayers object
        combinator: function with signature (z_c, u, size)
        encoder_layers: seq of ints
        denoising_cost: seq of floats
        batch_size: int


    Attributes
    ----------
        z_est: dict of tensors
        d_cost: seq of scalar tensors

    """

    def __init__(self, clean, corr, bn, combinator, encoder_layers,
                 denoising_cost, batch_size=100, scope='dec', reuse=None):


        self.clean = clean
        self.corr = corr
        self.bn = bn
        self.combinator = combinator
        self.encoder_layers = encoder_layers
        self.denoising_cost = denoising_cost
        self.batch_size = batch_size
        self.scope = scope
        self.reuse = reuse

        self.num_layers = len(encoder_layers) - 1

        self.z_est = {}  # activation reconstruction
        self.d_cost = []  # denoising cost

        self.build()

    def build(self):
        for l in range(self.num_layers, -1, -1):
            self.d_cost.append(self.create_layer(l))


    def create_layer(self, l):
        ls = self.encoder_layers
        denoising_cost = self.denoising_cost
        batch_size = self.batch_size
        clean = self.clean
        corr = self.corr
        bn = self.bn
        combinator = self.combinator
        num_layers = self.num_layers
        scope = self.scope
        reuse = self.reuse

        join, split_lu, labeled, unlabeled = get_batch_ops(batch_size)
        if VERBOSE:
            print("Layer {}: {} -> {}, denoising cost: {}".format(
            l, ls[l+1] if l + 1 < len(ls) else None,
            ls[l], denoising_cost[l]
        ))

        z, z_c = clean.unlabeled.z[l], corr.unlabeled.z[l]
        m, v = clean.unlabeled.m.get(l, 0), \
               clean.unlabeled.v.get(l, 1 - 1e-10)

        if l == num_layers:
            u = unlabeled(corr.logits)
        else:
            u = layers.fully_connected(
                self.z_est[l + 1],
                num_outputs=ls[l],
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(
                    stddev=1 / math.sqrt(ls[l + 1])),
                biases_initializer=None,
                scope=scope + str(l),
                reuse=reuse
            )

        u = bn.batch_normalization(u)

        with tf.variable_scope('cmb' + str(l), reuse=None):
            self.z_est[l] = combinator(z_c, u, ls[l])

        z_est_bn = (self.z_est[l] - m) / v

        # append the cost of this layer to d_cost
        reduce_axes = list(range(1, len(z_est_bn.get_shape().as_list())))

        d_cost = (tf.reduce_mean(
            tf.reduce_sum(
                tf.square(z_est_bn - z),
                axis=reduce_axes
            )) / ls[l]) * denoising_cost[l]

        return d_cost


# Gamma Decoder
class GammaDecoder(Decoder):
    def build(self):
        d_cost = self.create_layer(self.num_layers)
        self.d_cost.append(d_cost)



# -----------------------------
# BATCH NORMALIZATION
# -----------------------------
class BatchNormLayers(object):
    """Batch norm class

    Arguments
    ---------
        ls: sequence of ints
        scope: str

    Attributes
    ----------
        bn_assigns: list of TF ops
        ema: TF op
        running_var: list of tensors
        running_mean: list of tensors
        beta: list of tensors
        gamma: list of tensors


    """

    def __init__(self, ls, decay=0.99):
        # store updates to be made to average mean, variance
        self.bn_assigns = []
        # calculate the moving averages of mean and variance
        self.ema = tf.train.ExponentialMovingAverage(decay=decay)

        # average mean and variance of all layers
        # shift & scale
        # with tf.variable_scope(scope, reuse=None):

        self.running_var = [tf.get_variable(
            'v' + str(i),
            initializer=tf.constant(1.0, shape=[l]),
            trainable=False) for i, l in enumerate(ls[1:])]

        self.running_mean = [tf.get_variable(
            'm' + str(i),
            initializer=tf.constant(0.0, shape=[l]),
            trainable=False) for i, l in enumerate(ls[1:])]

        # shift
        self.beta = [tf.get_variable(
            'beta' + str(i),
            initializer=tf.constant(0.0, shape=[l])
        ) for i, l in enumerate(ls[1:])]

        # scale
        self.gamma = [tf.get_variable(
            'gamma' + str(i),
            initializer=tf.constant(1.0, shape=[l]))
            for i, l in enumerate(ls[1:])]

    def update_batch_normalization(self, batch, l):
        """
        batch normalize + update average mean and variance of layer l
        if CNN, use channel-wise batch norm
        """
        # bn_axes = [0, 1, 2] if self.params.cnn else [0]
        bn_axes = list(range(len(batch.get_shape().as_list())-1))
        # bn_axes = [0]
        mean, var = tf.nn.moments(batch, axes=bn_axes)
        assign_mean = self.running_mean[l - 1].assign(mean)
        assign_var = self.running_var[l - 1].assign(var)
        self.bn_assigns.append(
            self.ema.apply([self.running_mean[l - 1], self.running_var[l - 1]]))

        with tf.control_dependencies([assign_mean, assign_var]):
            return tf.nn.batch_normalization(batch, mean, var, offset=None,
                                             scale=None, variance_epsilon=1e-10)

    @staticmethod
    def batch_normalization(batch, mean=None, var=None):
        # bn_axes = [0, 1, 2] if self.params.cnn else [0]
        bn_axes = list(range(len(batch.get_shape().as_list())-1))
        # bn_axes = [0]

        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=bn_axes)

        # return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
        return tf.nn.batch_normalization(batch, mean, var, offset=None,
                                         scale=None, variance_epsilon=1e-10)


# -----------------------------
# COMBINATOR
# -----------------------------
def gauss_combinator(z_c, u, size):
    "gaussian denoising function proposed in the original paper"

    # wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    def wi(inits, name):
        return tf.get_variable(name, initializer=inits * tf.ones([size]))

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


class Model(object):
    """
    Base class for models.

    Arguments
    ---------
        inputs
        outputs
        train_flag
        params
        bn

    Required Attributes
    -------------------
        bn
        bn_decay
        clean
        u_cost
        cost
        predict

    Additional attributes (ladder)
    ------------------------------
        corr
        dec

    Additional attributes (vat)
    ---------------------------
        adv

    """

    def __init__(self, inputs, outputs, train_flag, params):
        self.params = params
        self.inputs = inputs
        self.outputs = outputs
        self.train_flag = train_flag

        # Supervised components
        if VERBOSE:
            print("=== Batch Norm === ")
        self.bn_decay = self.params.static_bn

        self.bn = BatchNormLayers(self.params.encoder_layers,
                                  decay=self.bn_decay)
        if VERBOSE:
            print("=== Clean Encoder ===")
        self.clean = self.get_encoder()

        self.num_layers = self.clean.num_layers

        # Compute predictions
        self.predict = tf.argmax(self.clean.logits, 1)

        self.build_unsupervised()
        self.cost = self.get_cost()
        self.u_cost = self.get_u_cost()

    def get_encoder(self):
        return Encoder(
            inputs=self.inputs, bn=self.bn, is_training=self.train_flag,
            params=self.params, this_encoder_noise=0.0,
            start_layer=0, update_batch_stats=True,
            scope='enc', reuse=None)

    def labeled(self, x):
        return x[:self.params.batch_size] if x is not None else x

    def unlabeled(self, x):
        return x[self.params.batch_size:] if x is not None else x

    def get_cost(self):
        # Calculate supervised cross entropy cost
        ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.outputs, logits=self.labeled(self.clean.logits))
        return tf.reduce_mean(ce)

    def get_u_cost(self):
        return None

    def build_unsupervised(self):
        return None


class VirtualAdversarialTraining(Model):

    def get_encoder(self):
        return VATEncoder(
            inputs=self.inputs, bn=self.bn, is_training=self.train_flag,
            params=self.params, this_encoder_noise=self.params.vadv_sd,
            start_layer=0, update_batch_stats=True,
            scope='enc', reuse=None
        )

    def build_unsupervised(self):
        self.adv = Adversary(
            self.bn, self.params, layer_eps=self.params.epsilon[0],
            start_layer=0)

    def get_u_cost(self):
        return self.adv.virtual_adversarial_loss(
            x=self.unlabeled(self.inputs),
            logit=self.unlabeled(self.clean.logits),
            is_training=self.train_flag)




class Ladder(Model):
    """"""
    def build_unsupervised(self):
        if VERBOSE:
            print("=== Corrupted Encoder === ")
        self.corr = self.get_corrupted_encoder()

        if VERBOSE:
            print("=== Decoder ===")
        self.dec = self.get_decoder()

    def get_u_cost(self):
        return tf.add_n(self.dec.d_cost)

    def get_cost(self):
        # Overrides base class since we want to use corrupted logits for cost
        # Calculate supervised cross entropy cost
        ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.outputs, logits=self.labeled(self.corr.logits))
        return tf.reduce_mean(ce)

    def get_corrupted_encoder(self):
        start_layer = 0
        update_batch_stats = False
        scope = 'enc'
        reuse = True
        params = self.params
        return Encoder(
            inputs=self.inputs, bn=self.bn, is_training=self.train_flag,
            params=self.params, this_encoder_noise=params.corrupt_sd,
            start_layer=start_layer, update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse)

    def get_decoder(self):
        return Decoder(
            clean=self.clean, corr=self.corr, bn=self.bn,
            combinator=gauss_combinator,
            encoder_layers=self.params.encoder_layers,
            denoising_cost=self.params.rc_weights,
            batch_size=self.params.batch_size,
            scope='dec', reuse=None)

#  Gamma
class Gamma(Ladder):

    def get_encoder(self):
        if self.params.cnn:
            return ConvEncoder(
                inputs=self.inputs, bn=self.bn, is_training=self.train_flag,
                params=self.params, this_encoder_noise=0.0,
                start_layer=0, update_batch_stats=True,
                scope='enc', reuse=None)
        else:
            super(Gamma, self).get_encoder()


    def get_corrupted_encoder(self):
        if self.params.cnn:
            start_layer = 0
            update_batch_stats = False
            scope = 'enc'
            reuse = True
            params = self.params
            return ConvEncoder(
                inputs=self.inputs, bn=self.bn, is_training=self.train_flag,
                params=params, this_encoder_noise=params.corrupt_sd,
                start_layer=start_layer, update_batch_stats=update_batch_stats,
                scope=scope, reuse=reuse)
        else:
            super(Gamma, self).get_corrupted_encoder()

    def get_decoder(self):
        return GammaDecoder(
            clean=self.clean, corr=self.corr, bn=self.bn,
            combinator=gauss_combinator,
            encoder_layers=self.params.encoder_layers,
            denoising_cost=self.params.rc_weights,
            batch_size=self.params.batch_size,
            scope='dec', reuse=None
        )

#  VAN encoder
class LadderWithVAN(Ladder):
    def get_corrupted_encoder(self):
        start_layer = 0
        update_batch_stats = False
        scope = 'enc'
        reuse = True
        params = self.params
        return VirtualAdversarialNoiseEncoder(
            self.inputs, self.bn, self.train_flag, params, self.clean.logits,
            this_encoder_noise=params.corrupt_sd,
            start_layer=start_layer, update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse)


# -----------------------------
# -----------------------------
# VAT FUNCTIONS
# -----------------------------
# -----------------------------
def ce_loss(logit, y):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


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


def get_normalized_vector(d):
    red_axes = list(range(1, len(d.get_shape())))
    # print(d.get_shape(), red_axes)
    d /= (1e-12 + tf.reduce_max(tf.abs(d), axis=red_axes,
                                keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0),
                                      axis=red_axes,
                                      keep_dims=True))
    return d


def softmax_cross_entropy_with_logits(labels, logits):
    q = tf.nn.softmax(labels)
    return -tf.reduce_mean(tf.reduce_sum(q * logsoftmax(logits), 1))


class Adversary(object):
    def __init__(self,
                 bn,
                 params,
                 layer_eps=5.0,
                 start_layer=0):
        # Ladder (encoder parameters)
        self.bn = bn
        self.params = params
        self.start_layer = start_layer

        # VAT
        self.this_adv_eps = layer_eps
        self.xi = params.xi
        self.num_power_iters = params.num_power_iters

    def forward(self, x, is_training, update_batch_stats=False):
        # always use a standard Gaussian-noise encoder
        vatfw = Encoder(
            inputs=x,
            bn=self.bn,
            is_training=is_training,
            params=self.params,  # for encoder_layers, batch_size, van settings
            this_encoder_noise=self.params.vadv_sd,
            # add gaussian for stability
            start_layer=self.start_layer,
            update_batch_stats=update_batch_stats,
            scope='enc', reuse=True)
        return vatfw.logits  # logits by default includes both labeled/unlabeled

    def generate_virtual_adversarial_perturbation(self, x, logit, is_training):
        if VERBOSE:
            print("--- VAT Pass: Generating VAT perturbation ---")
        d = tf.random_normal(shape=tf.shape(x))
        for k in range(self.num_power_iters):
            d = self.xi * get_normalized_vector(d)
            logit_p = logit
            if VERBOSE:
                print("Power Iteration: {}".format(k))
            logit_m = self.forward(x + d, update_batch_stats=False,
                                   is_training=is_training)
            dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
        return self.this_adv_eps * get_normalized_vector(d)

    def generate_adversarial_perturbation(self, x, loss):
        grad = tf.gradients(loss, [x], aggregation_method=2)[0]
        grad = tf.stop_gradient(grad)
        return self.this_adv_eps * get_normalized_vector(grad)

    def adversarial_loss(self, x, y, loss, is_training,
                         name="at_loss"):
        r_adv = self.generate_adversarial_perturbation(x, loss)
        logit = self.forward(x + r_adv, is_training=is_training,
                             update_batch_stats=False)
        loss = ce_loss(logit, y)
        return tf.identity(loss, name=name)

    def virtual_adversarial_loss(self, x, logit, is_training,
                                 name="vat_loss"):
        r_vadv = self.generate_virtual_adversarial_perturbation(
            x, logit, is_training=is_training)
        logit = tf.stop_gradient(logit)
        logit_p = logit
        if VERBOSE:
            print("--- VAT Pass: Computing VAT Loss (KL Divergence) ---")
        logit_m = self.forward(x + r_vadv, update_batch_stats=False,
                               is_training=is_training)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return tf.identity(loss, name=name)

#

class VATAdversary(Adversary):
    def __init__(self, params):
        super(VATAdversary, self).__init__(
            bn=None, params=params, layer_eps=params.epsilon[0], start_layer=0)


    def vat_bn(self, x, dim, is_training=True, update_batch_stats=True,
               collections=None,
               name="bn"):
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
                    self.params.static_bn * avg_mean + (
                        1 - self.params.static_bn) * mean)
                avg_var_assign_op = tf.assign(
                    avg_var,
                    self.params.static_bn * avg_var + (n / (n - 1))
                    * (1 - self.params.static_bn) * var)

            with tf.control_dependencies(
                    [avg_mean_assign_op, avg_var_assign_op]):
                z = (x - mean) / tf.sqrt(1e-6 + var)
        else:
            z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

        return gamma * z + beta

    def logit(self, x, is_training=True, update_batch_stats=True):

        params = self.params

        def weight(s, i):
            return tf.get_variable('w' + str(i), shape=s,
                                   initializer=tf.random_normal_initializer(
                                       stddev=(
                                           1 / math.sqrt(sum(s)))))

        def bias(s, i):
            return tf.get_variable('b' + str(i), shape=s,
                                   initializer=tf.zeros_initializer(dtype=tf.float32), dtype=tf.float32)

        ls = list(zip(params.encoder_layers[:-1], params.encoder_layers[1:]))

        h = x
        for i, l in enumerate(ls):
            h = lrelu(tf.matmul(h, weight(l, i)) + bias(l[-1], i), self.params.lrelu_a)
            if i < len(ls) - 1:
                h = self.vat_bn(h, l[-1], is_training=is_training,
                                update_batch_stats=update_batch_stats,
                                name='bn' + str(i))
                if is_training:  # for stabilisation
                    h += tf.random_normal(tf.shape(h), stddev=self.params.vadv_sd)

        return h

    def forward(self, x, is_training, update_batch_stats=False):

        def training_logit():
            return self.logit(x, is_training=True,
                              update_batch_stats=update_batch_stats)

        def testing_logit():
            return self.logit(x, is_training=False,
                              update_batch_stats=update_batch_stats)

        return tf.cond(is_training, training_logit, testing_logit)



def get_vat_cost(model, train_flag, params):
    def unlabeled(x):
        return x[params.batch_size:] if x is not None else x

    def get_layer_vat_cost(l):

        adv = Adversary(bn=model.bn,
                        params=params,
                        layer_eps=params.epsilon[l],
                        start_layer=l)

        # VAT on unlabeled only
        return (
            adv.virtual_adversarial_loss(
                x=model.corr.unlabeled.z[l],
                logit=unlabeled(model.corr.logits),  # should this be clean?
                is_training=train_flag)
        )

    if params.model == "clw":
        vat_costs = []
        for l in range(model.num_layers):
            vat_costs.append(get_layer_vat_cost(l))
        vat_cost = tf.add_n(vat_costs)

    elif params.model == "c":
        vat_cost = get_layer_vat_cost(0)

    else:
        vat_cost = 0.0

    return vat_cost




# -----------------------------
# -----------------------------

def build_graph(params):

    # -----------------------------
    # Placeholder setup
    inputs_placeholder = tf.placeholder(
        tf.float32, shape=(None, params.input_size))

    inputs = preprocess(inputs_placeholder, params)
    outputs = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    if params.model == "c" or params.model == "clw":
        model = Ladder(inputs, outputs, train_flag, params)
        vat_cost = get_vat_cost(model, train_flag, params)
        loss = model.cost + model.u_cost + vat_cost
        s_cost = model.cost
        u_cost = model.u_cost

    elif params.model == "n" or params.model == "nlw":
        model = LadderWithVAN(inputs, outputs, train_flag, params)
        vat_cost = tf.zeros([])
        loss = model.cost + model.u_cost
        s_cost = model.cost
        u_cost = model.u_cost

    elif params.model == "vat":
        model = VATAdversary(params)
        logit = model.forward(x=inputs[:params.batch_size], is_training=train_flag, update_batch_stats=True)
        s_cost = ce_loss(logit=logit,y=outputs)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            ul_x = inputs[params.batch_size:]
            ul_logit = model.forward(ul_x, is_training=train_flag,
                                     update_batch_stats=False)
            vat_cost = model.virtual_adversarial_loss(ul_x, ul_logit)
            loss = s_cost + vat_cost
            u_cost = vat_cost

        # model = VirtualAdversarialTraining(inputs, outputs, train_flag, params)
        # vat_cost = model.u_cost
        # loss = model.cost + model.u_cost

    elif params.model == "gamma":
        model = Gamma(inputs, outputs, train_flag, params)
        vat_cost = tf.zeros([])
        loss = model.cost + model.u_cost
        s_cost = model.cost
        u_cost = model.u_cost

    else:
        model = Ladder(inputs, outputs, train_flag, params)
        vat_cost = tf.zeros([])
        loss = model.cost + model.u_cost
        s_cost = model.cost
        u_cost = model.u_cost

    # -----------------------------
    # Loss, accuracy and training steps

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(model.predict, tf.argmax(outputs, 1)),
            "float")) * tf.constant(100.0)

    # learning_rate = tf.Variable(params.initial_learning_rate, name='lr', trainable=False)
    # beta1 = tf.Variable(params.beta1, name='beta1', trainable=False)
    learning_rate = tf.placeholder_with_default(params.initial_learning_rate, shape=[], name='lr')
    beta1 = tf.placeholder_with_default(params.beta1, shape=[], name='beta1')

    train_step = tf.train.AdamOptimizer(learning_rate,
                                        beta1=beta1).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*model.bn.bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5,
                           max_to_keep=5)

    # Graph
    g = dict()
    g['images'] = inputs_placeholder
    g['labels'] = outputs
    g['train_flag'] = train_flag
    g['ladder'] = model
    g['saver'] = saver
    g['train_step'] = train_step
    g['lr'] = learning_rate
    g['beta1'] = beta1

    # Metrics
    m = dict()
    m['loss'] = loss
    m['cost'] = s_cost
    m['uc'] = u_cost
    m['acc'] = accuracy
    m['vc'] = vat_cost

    trainable_params = count_trainable_params()

    return g, m, trainable_params


def get_spectral_radius(x, logit, forward, num_power_iters=1, xi=1e-6):
    prev_d = tf.random_normal(shape=tf.shape(x))
    for k in range(num_power_iters):
        d = xi * get_normalized_vector(prev_d)
        logit_p = logit
        logit_m = forward(x + d)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        prev_d = tf.stop_gradient(grad)

    prev_d, d = get_normalized_vector(prev_d), get_normalized_vector(d)

    def dot(a, b): return tf.reduce_mean(tf.multiply(a, b), axis=1)

    return dot(d, prev_d) / dot(prev_d, prev_d)


def measure_smoothness(g, params):
    # Measure smoothness using clean logits
    if VERBOSE:
        print("=== Measuring smoothness ===")
    inputs = g['images']
    logits = g['ladder'].clean.logits

    def forward(x):
        return Encoder(
            inputs=x,
            bn=g['ladder'].bn,
            is_training=g['train_flag'],
            params=params,
            this_encoder_noise=0.0,
            start_layer=0,
            update_batch_stats=False,
            scope='enc',
            reuse=True).logits

    return get_spectral_radius(
        x=inputs, logit=logits, forward=forward, num_power_iters=5)
