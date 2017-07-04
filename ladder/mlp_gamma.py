import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import OrderedDict

class NoisyBNLayer(object):

    def __init__(self, scope_name, size, decay=0.99, var_ep=1e-5):
        self.scope_name = scope_name
        self.decay = decay
        self.var_ep = var_ep
        with tf.variable_scope(scope_name, reuse=False):
            self.scale = tf.get_variable('NormScale', initializer=tf.ones([size]))
            self.beta = tf.get_variable('NormOffset', initializer=tf.zeros([size]))
            self.pop_mean = tf.get_variable('PopulationMean', initializer=tf.zeros([size]), trainable=False)
            self.pop_var = tf.get_variable('PopulationVariance', initializer=tf.fill([size], 1e-2), trainable=False)
        # self.batch_mean, self.batch_var = None, None

    def normalize(self, x, training=True):
        eps = self.var_ep
        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0])
            # self.batch_mean, self.batch_var = batch_mean, batch_var
            train_mean_op = tf.assign(self.pop_mean,
                                      self.pop_mean * self.decay + batch_mean * (1 - self.decay))
            train_var_op = tf.assign(self.pop_var,
                                     self.pop_var * self.decay + batch_var * (1 - self.decay))

            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.divide(tf.subtract(x, batch_mean), tf.add(tf.sqrt(batch_var), eps))

        else:
            return tf.divide(tf.subtract(x, self.pop_mean), tf.add(tf.sqrt(self.pop_var), eps))


    def apply(self, x, training=True, noise=None, scale=True):
        x_normed = self.normalize(x, training=training)
        if noise is not None:
            x_normed += noise

        if scale:
            return tf.multiply(self.scale, tf.add(x_normed, self.beta))
        else:
            return tf.add(x_normed, self.beta)


class Encoder(object):
    def __init__(self, scope, x, y, layer_sizes, noise_sd=None):
        self.scope = scope
        self.n_layers = len(layer_sizes)
        self.layer_sizes = {i: layer_sizes[i] for i in range(self.n_layers)}

        self.x = x
        self.z_pre = OrderedDict()
        self.z = OrderedDict()
        self.h = OrderedDict()
        self.y = y

        if noise_sd is None:
            self.noise = {i: None for i in range(self.n_layers)}
            self.h[0] = x
        else:
            self.noise = {i: tf.random_normal(shape=(BATCH_SIZE, layer_sizes[i]), mean=0.0, stddev=noise_sd)
                          for i in range(self.n_layers)}
            self.h[0] = x + self.noise[0]

        self.bn_layers = {i: NoisyBNLayer(scope + str(i), enc_layers[i]) for i in range(self.n_layers)}

        self.s_cost = None


        self.wts_init = layers.xavier_initializer()
        self.bias_init = tf.truncated_normal_initializer(stddev=1e-6)

    def forward_pass(self, training=True):
        for l in range(1, self.n_layers):
            output_size = self.layer_sizes[l]
            bn = self.bn_layers[l]

            self.z_pre[l] = layers.fully_connected(
                inputs=self.h[l - 1],
                num_outputs=output_size,
                activation_fn=None,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=self.wts_init,
                weights_regularizer=None,
                biases_initializer=self.bias_init,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=self.scope + str(l)
            )

            if l == self.n_layers - 1:
                self.z[l] = bn.apply(x=self.z_pre[l], training=training, noise=noise[l], scale=True)
                self.h[l] = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=self.z[l])
            else:
                self.z[l] = bn.apply(x=self.z_pre[l], training=training, noise=self.noise[l], scale=False)
                self.h[l] = tf.nn.relu(self.z[l])

        self.s_cost = self.h[-1]


BATCH_SIZE = 100
INPUT_SIZE = 784
TRAIN_FLAG = True
OUTPUT_SIZE = 10

# ===========================
# ENCODER
# ===========================

# Start with a tuple specifying layer sizes
enc_layers = [INPUT_SIZE, 1000, 500, 250, 250, 250, OUTPUT_SIZE]

# Input placeholder
x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_SIZE))
# One-hot targets
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_SIZE))

# ===========================
# CLEAN ENCODER
clean_encoder = Encoder('clean', x, y, enc_layers, noise_sd=None)
clean_encoder.forward_pass(TRAIN_FLAG)

# ===========================
# CORRUPTED ENCODER
noisy_encoder = Encoder('corrupted', x, y, enc_layers, noise_sd=0.3)
noisy_encoder.forward_pass(TRAIN_FLAG)


# ===========================
# ===========================
# DECODER
# ===========================
# ===========================

class DecoderLayer(object):
    def __init__(self, input_size, output_size):
        



