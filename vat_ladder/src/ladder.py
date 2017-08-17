from tensorflow.contrib import layers as layers
import tensorflow as tf
import math

# -----------------------------
# MODEL BUILDING
# -----------------------------

def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha*x)

def get_batch_ops(batch_size):
    join = lambda l, u: tf.concat([l, u], 0)
    split_lu = lambda x: (labeled(x), unlabeled(x))
    labeled = lambda x: x[:batch_size] if x is not None else x
    unlabeled = lambda x: x[batch_size:] if x is not None else x
    return join, split_lu, labeled, unlabeled

def preprocess(placeholder, params):
    return tf.reshape(placeholder, shape=[
        -1, params.cnn_init_size, params.cnn_init_size, params.cnn_fan[0]
    ]) if params.cnn else placeholder


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
        encoder_layers: sequence of ints
        bn: BatchNormLayers
        is_training: tensorflow bool
        noise_sd: float
        start_layer: int
        batch_size: int
        update_batch_stats: bool

    Attributes
    ----------
        logits, pre-softmax output at final layer
        labeled, an Activations object with attributes z, h, m, v
        unlabeled, an Activations object


    """
    def __init__(self,
                 inputs,
                 encoder_layers,
                 bn,
                 is_training,
                 noise_sd=0.0,
                 start_layer=0,
                 batch_size=100,
                 update_batch_stats=True,
                 scope='enc',
                 reuse=None):

        self.noise_sd = noise_sd
        self.labeled = Activations()
        self.unlabeled = Activations()
        join, split_lu, labeled, unlabeled = get_batch_ops(batch_size)

        ls = encoder_layers  # seq of layer sizes, len num_layers
        self.num_layers = len(encoder_layers) - 1

        # Layer 0: inputs, size 784
        l = 0
        h = inputs + self.generate_noise(inputs, l)
        self.labeled.z[l], self.unlabeled.z[l] = split_lu(h)

        for l in range(start_layer+1, self.num_layers + 1):
            print("Layer {}: {} -> {}".format(l, ls[l - 1], ls[l]))
            self.labeled.h[l-1], self.unlabeled.z[l-1] = split_lu(h)
            # z_pre = tf.matmul(h, self.W[l-1])
            z_pre = layers.fully_connected(
                h,
                num_outputs=ls[l],
                weights_initializer=tf.random_normal_initializer(
                    stddev=1/math.sqrt(ls[l-1])),
                biases_initializer=None,
                activation_fn=None,
                scope=scope+str(l),
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
                    noise = self.generate_noise(z_pre, l)
                    z += noise
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance
                    # using batch mean and variance of labeled examples
                    bn_l = bn.update_batch_normalization(z_pre_l, l) if \
                        update_batch_stats else bn.batch_normalization(z_pre_l)
                    bn_u = bn.batch_normalization(z_pre_u, m, v)
                    z = join(bn_l, bn_u)
                return z

            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = bn.ma.average(bn.running_mean[l - 1])
                var = bn.ma.average(bn.running_var[l - 1])
                z = bn.batch_normalization(z_pre, mean, var)

                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(is_training, training_batch_norm, eval_batch_norm)

            if l == self.num_layers:
                # return pre-softmax logits in final layer
                self.logits = bn.gamma[l-1] * (z + bn.beta[l-1])
                h = tf.nn.softmax(self.logits)
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + bn.beta[l - 1])
                # h = lrelu(z + bn.beta[l-1])

            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l], self.unlabeled.v[l] = m, v
            self.labeled.z[l], self.unlabeled.z[l] = split_lu(z)
            self.labeled.h[l], self.unlabeled.h[l] = split_lu(h)


    def generate_noise(self, inputs, l):
        """Add noise depending on corruption parameters"""
        # start_layer = l+1
        # corrupt = self.params.corrupt
        # if corrupt == 'vatgauss':
        #     noise = generate_virtual_adversarial_perturbation(
        #         inputs, clean_logits, is_training=is_training,
        #         start_layer=start_layer) + \
        #         tf.random_normal(tf.shape(inputs)) * noise_sd
        # elif corrupt == 'vat':
        #     noise = generate_virtual_adversarial_perturbation(
        #         inputs, clean_logits, is_training=is_training,
        #         start_layer=start_layer)
        # elif corrupt == 'gauss':
        #     noise = tf.random_normal(tf.shape(inputs)) * noise_sd
        # else:
        #     noise = tf.zeros(tf.shape(inputs))
        # return noise
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
            with tf.variable_scope('cmb' + str(l), reuse=None):
                z_est[l] = combinator.combine(z_c, u, ls[l], l)

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
        ma: TF op
        running_var: list of tensors
        running_mean: list of tensors
        beta: list of tensors
        gamma: list of tensors


    """
    def __init__(self, ls, bn_decay, scope='bn'):

        # store updates to be made to average mean, variance
        self.bn_assigns = []
        # calculate the moving averages of mean and variance
        self.ma = tf.train.ExponentialMovingAverage(decay=bn_decay)

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
            self.ma.apply([self.running_mean[l - 1], self.running_var[l - 1]]))

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

class Combinator(object):
    """Combinator function factory, augmented MLP combinator by default"""
    def __init__(self, layers=(4,1), init_sd=0.025):
        print(self.init_statement)
        layers = [self.input_size] + layers
        self.shapes = list(zip(layers[:-1], layers[1:]))

        self.w = lambda n_in, n_out, name: tf.get_variable(
                'w'+name,
                 shape=[n_in, n_out],
                 initializer=tf.random_normal_initializer(
                     stddev=init_sd
                 )
            )

        self.b = lambda n_out, name: tf.get_variable(
                'b'+name,
                shape=[n_out],
                initializer=tf.zeros_initializer(dtype="float"))

    def preprocess(self, z_c, u):
        uz = tf.multiply(z_c, u)
        return tf.concat([tf.reshape(z, [-1, 1]) for z in [z_c, u, uz]],
                         axis=-1)

    def combine(self, z_c, u, size, l):
        print("Combining at layer", l)
        h = self.preprocess(z_c, u)
        target_shape = [x if x is not None else -1 for x in z_c.get_shape().as_list()]

        for i, shape in enumerate(self.shapes):
            h = lrelu(tf.nn.xw_plus_b(h,
                                      weights=self.w(*shape, name=str(i)),
                                      biases=self.b(shape[1], name=str(i))))

        return tf.reshape(h, target_shape)

    @property
    def init_statement(self):
        return "=== AMLP Combinator ==="

    @property
    def input_size(self):
        return 3

class MLPCombinator(Combinator):
    def preprocess(self, z_c, u):
        return tf.concat([tf.reshape(z, [-1, 1]) for z in [z_c, u]],
                         axis=-1)
    @property
    def init_statement(self):
        return "=== MLP Combinator ==="

    @property
    def input_size(self):
        return 2

class GaussCombinator(Combinator):
    @property
    def init_statement(self):
        return "=== Gauss (Vanilla) Combinator ==="

    def combine(self, z_c, u, size, l):
        """ Gaussian assumption on z: (z - mu) * v + mu"""

        wi = lambda inits, name: tf.get_variable(
            name,
            initializer=tf.ones_initializer(dtype="float"),
            shape=[size])

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
