import tensorflow as tf
import tensorflow.contrib.layers as layers
import math
from src.utils import get_batch_ops

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
        self.z = {} # pre-activation
        self.h = {} # activation
        self.m = {} # mean
        self.v = {} # variance

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
        noise_sd: float
        params: argparse Namespace object with attributes:
            encoder_layers: sequence of ints
            batch_size: int
        start_layer: int
        update_batch_stats: bool
        scope: str
        reuse: bool or None

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

        self.bn = bn
        self.start_layer = start_layer
        self.is_training = is_training
        el = params.encoder_layers
        self.encoder_layers = el
        self.batch_size = params.batch_size
        self.noise_sd = this_encoder_noise

        self.labeled = Activations()
        self.unlabeled = Activations()
        join, split_lu, labeled, unlabeled = get_batch_ops(self.batch_size)

        # encoder_layers = encoder_layers  # seq of layer sizes, len num_layers
        self.num_layers = len(el) - 1

        # Layer 0: inputs, size 784
        h = inputs + self.generate_noise(inputs, start_layer)
        self.labeled.z[start_layer], self.unlabeled.z[start_layer] = split_lu(h)
        if start_layer +1 > self.num_layers:
            print(inputs.get_shape())

        for l_out in range(start_layer+1, self.num_layers+1):
            l_in = l_out-1
            self.print_progress(l_out)

            self.labeled.h[l_in], self.unlabeled.z[l_in] = split_lu(h)
            # z_pre = tf.matmul(h, self.W[l-1])
            z_pre = layers.fully_connected(
                h,
                num_outputs=el[l_out],
                weights_initializer=tf.random_normal_initializer(
                    stddev=1/math.sqrt(el[l_in])),
                biases_initializer=None,
                activation_fn=None,
                scope=scope+str(l_out),
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
                self.logits = bn.gamma[l_in] * (z + bn.beta[l_in])
                h = tf.nn.softmax(self.logits)
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + bn.beta[l_in])

            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l_out], self.unlabeled.v[l_out] = m, v
            self.labeled.z[l_out], self.unlabeled.z[l_out] = split_lu(z)
            self.labeled.h[l_out], self.unlabeled.h[l_out] = split_lu(h)

    def print_progress(self, l_out):
        el = self.encoder_layers
        print("Layer {}: {} -> {}".format(l_out, el[l_out - 1], el[l_out]))

    def generate_noise(self, inputs, l_out):
        return tf.random_normal(tf.shape(inputs)) * self.noise_sd

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

        # self.params = params
        ls = encoder_layers  # seq of layer sizes, len num_layers
        num_layers = len(encoder_layers) - 1
        # denoising_cost = params.rc_weights
        join, split_lu, labeled, unlabeled = get_batch_ops(batch_size)
        z_est = {}  # activation reconstruction
        d_cost = []  # denoising cost

        for l in range(num_layers, -1, -1):
            print("Layer {}: {} -> {}, denoising cost: {}".format(
                l, ls[l+1] if l+1<len(ls) else None,
                ls[l], denoising_cost[l]
            ))

            z, z_c = clean.unlabeled.z[l], corr.unlabeled.z[l]
            m, v = clean.unlabeled.m.get(l, 0), \
                   clean.unlabeled.v.get(l, 1-1e-10)
            # print(l)
            if l == num_layers:
                u = unlabeled(corr.logits)
            else:
                u = layers.fully_connected(
                    z_est[l+1],
                    num_outputs=ls[l],
                    activation_fn=None,
                    normalizer_fn=None,
                    weights_initializer=tf.random_normal_initializer(
                        stddev=1/math.sqrt(ls[l+1])),
                    biases_initializer=None,
                    scope=scope+str(l),
                    reuse=reuse
                    )

            u = bn.batch_normalization(u)

            with tf.variable_scope('cmb'+str(l), reuse=None):
                z_est[l] = combinator(z_c, u, ls[l])

            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            reduce_axes = list(range(1,len(z_est_bn.get_shape().as_list())))
            d_cost.append((tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(z_est_bn - z),
                    axis=reduce_axes
                    )) / ls[l]) * denoising_cost[l])

        self.z_est = z_est
        self.d_cost = d_cost

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
            'v'+str(i),
            initializer=tf.constant(1.0, shape=[l]),
            trainable=False) for i,l in enumerate(ls[1:])]

        self.running_mean = [tf.get_variable(
            'm'+str(i),
            initializer=tf.constant(0.0, shape=[l]),
            trainable=False) for i,l in enumerate(ls[1:])]

        # shift
        self.beta = [tf.get_variable(
            'beta'+str(i),
            initializer=tf.constant(0.0, shape=[l])
        ) for i,l in enumerate(ls[1:])]

        # scale
        self.gamma = [tf.get_variable(
            'gamma'+str(i),
            initializer=tf.constant(1.0, shape=[l]))
            for i,l in enumerate(ls[1:])]


    def update_batch_normalization(self, batch, l):
        """
        batch normalize + update average mean and variance of layer l
        if CNN, use channel-wise batch norm
        """
        # bn_axes = [0, 1, 2] if self.params.cnn else [0]
        # bn_axes = list(range(len(batch.get_shape().as_list())-1))
        bn_axes = [0]
        mean, var = tf.nn.moments(batch, axes=bn_axes)
        assign_mean = self.running_mean[l-1].assign(mean)
        assign_var = self.running_var[l-1].assign(var)
        self.bn_assigns.append(
            self.ema.apply([self.running_mean[l-1], self.running_var[l-1]]))

        with tf.control_dependencies([assign_mean, assign_var]):
            return tf.nn.batch_normalization(batch, mean, var, offset=None,
                                             scale=None, variance_epsilon=1e-10)

    def batch_normalization(self, batch, mean=None, var=None):
        # bn_axes = [0, 1, 2] if self.params.cnn else [0]
        # bn_axes = list(range(len(batch.get_shape().as_list())-1))
        bn_axes = [0]

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
    wi = lambda inits, name: tf.get_variable(name,
                                             initializer=inits*tf.ones([size]))
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




class Ladder(object):
    """"""
    def __init__(self, inputs, outputs, train_flag, params):
        """

        :param inputs: tensor or placeholder
        :param outputs:
        :param train_flag:
        :param params:
        """

        # join, split_lu, labeled, unlabeled = get_batch_ops(params.batch_size)
        labeled = lambda x: x[:params.batch_size] if x is not None else x

        print("=== Batch Norm === ")
        if params.static_bn is False:
            self.bn_decay = tf.Variable(1e-10, trainable=False)
        else:
            self.bn_decay = params.static_bn

        self.bn = BatchNormLayers(params.encoder_layers, decay=self.bn_decay)

        print("=== Clean Encoder ===")
        self.clean = Encoder(
            inputs=inputs, bn=self.bn, is_training=train_flag,
            params=params, this_encoder_noise=0.0,
            start_layer=0, update_batch_stats=True,
            scope='enc', reuse=None)

        print("=== Corrupted Encoder === ")
        self.corr = self.get_corrupted_encoder(
            inputs, self.bn, train_flag, params)

        print("=== Decoder ===")
        self.dec = Decoder(
            clean=self.clean, corr=self.corr, bn=self.bn,
            combinator=gauss_combinator,
            encoder_layers=params.encoder_layers,
            denoising_cost=params.rc_weights,
            batch_size=params.batch_size,
            scope='dec', reuse=None)


        self.num_layers = self.clean.num_layers

        # Calculate total unsupervised cost
        self.u_cost = tf.add_n(self.dec.d_cost)

        # Calculate supervised cross entropy cost
        ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=outputs, logits=labeled(self.corr.logits))
        self.cost = tf.reduce_mean(ce)

        # Compute predictions
        self.predict = tf.argmax(self.clean.logits, 1)


    def get_corrupted_encoder(self, inputs, bn, train_flag, params,
                              start_layer=0, update_batch_stats=False,
                              scope='enc', reuse=True):
        return Encoder(
            inputs=inputs, bn=bn, is_training=train_flag,
            params=params, this_encoder_noise=params.corrupt_sd,
            start_layer=start_layer, update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse)

