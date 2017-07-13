import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import OrderedDict



class NoisyBNLayer(object):

    def __init__(self, scope_name, size, noise_sd=None, decay=0.99, var_ep=1e-5, reuse=None):
        self.scope = scope_name
        self.size = size
        self.noise_sd = noise_sd
        self.decay = decay
        self.var_ep = var_ep
        with tf.variable_scope(scope_name, reuse=reuse):
            self.scale = tf.get_variable('NormScale', initializer=tf.ones([size]))
            self.beta = tf.get_variable('NormOffset', initializer=tf.zeros([size]))
            self.pop_mean = tf.get_variable('PopulationMean', initializer=tf.zeros([size]), trainable=False)
            self.pop_var = tf.get_variable('PopulationVariance', initializer=tf.fill([size], 1e-2), trainable=False)
        self.batch_mean, self.batch_var = None, None

    def normalize(self, x, training=True):
        eps = self.var_ep
        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0])
            self.batch_mean, self.batch_var = batch_mean, batch_var
            train_mean_op = tf.assign(self.pop_mean,
                                      self.pop_mean * self.decay + batch_mean * (1 - self.decay))
            train_var_op = tf.assign(self.pop_var,
                                     self.pop_var * self.decay + batch_var * (1 - self.decay))

            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.divide(tf.subtract(x, batch_mean), tf.add(tf.sqrt(batch_var), eps))

        else:
            return tf.divide(tf.subtract(x, self.pop_mean), tf.add(tf.sqrt(self.pop_var), eps))

    def normalize_from_saved_stats(self, x, training=True):
        if training:
            return tf.divide(tf.subtract(x, self.batch_mean), tf.add(tf.sqrt(self.batch_var), self.var_ep))
        else:
            return tf.divide(tf.subtract(x, self.pop_mean), tf.add(tf.sqrt(self.pop_var), self.var_ep))

    def add_noise(self, x):
        if self.noise_sd is not None:
            noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=self.noise_sd)
            return x + noise
        else:
            return x

    def apply_shift_scale(self, x, shift=True, scale=True):
        if shift:
            x = tf.add(x, self.beta)
        if scale:
            x = tf.multiply(self.scale, x)
        return x


class Encoder(object):
    def __init__(self, x, y, layer_sizes, noise_sd=None, reuse=None, training=True):
        """

        :param scope:
        :param x:
        :param y:
        :param layer_sizes:
        :param noise_sd: only a single scalar for all levels at this point
        """
        self.scope = 'enc'
        self.reuse = reuse
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.last = self.n_layers-1

        self.x = x
        self.z_pre = OrderedDict()
        self.z = OrderedDict()
        self.h = OrderedDict()
        self.y = y


        self.bn_layers = [NoisyBNLayer(scope_name=self.scope+'_bn'+str(i),
                                       size=layer_sizes[i],
                                       noise_sd=noise_sd,
                                       reuse=reuse) for i in range(self.n_layers)]


        self.predict = self.build(training)


    def build(self, training=True):
        """D
        Defines all operations needed for inference.
        Supervised loss also required for training.
        """
        bn = self.bn_layers[0]
        self.h[0] = bn.add_noise(bn.normalize(self.x, training))

        for l in range(1, self.n_layers):
            # print('enc', l)
            size_out = self.layer_sizes[l]
            bn = self.bn_layers[l]
            self.z_pre[l] = fclayer(self.h[l-1], size_out, reuse=self.reuse, scope=self.scope+str(l))

            if l == self.n_layers - 1:
                self.z[l] = bn.add_noise(bn.normalize(self.z_pre[l], training))
                self.h[l] = tf.nn.softmax(logits=bn.apply_shift_scale(self.z[l], shift=True, scale=True))
            else:
                # Do not need to apply scaling to RELU
                self.z[l] = bn.add_noise(bn.normalize(self.z_pre[l], training))
                self.h[l] = tf.nn.relu(bn.apply_shift_scale(self.z[l], shift=True, scale=False))


        predict = tf.argmax(self.h[self.n_layers-1], axis=-1)

        return predict

    def supervised_loss(self, labeled_batch_size, unlabeled_batch_size, training=True):
        labeled_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.z[self.n_layers - 1][:labeled_batch_size])
        if training:
            return tf.concat((labeled_loss, tf.zeros((unlabeled_batch_size,))), axis=0)
        else:
            return labeled_loss



class Combinator(object):
    def __init__(self, decoder_input, lateral_input, layer_sizes=(2, 2, 2), stddev=0.006, scope='com'):
        """
        :param inputs:
        :param layer_sizes: Hidden layers
        :param stddev: Standard deviation of weight initializer
        """
        self.wts_init = tf.random_normal_initializer(stddev=stddev)
        self.bias_init = tf.truncated_normal_initializer(stddev=1e-6)
        self.reuse = None
        self.scope = scope
        inputs = self.stack_input(decoder_input, lateral_input)
        self.outputs = self.build(inputs, layer_sizes)


    def build(self, inputs, layer_sizes):
        last = len(layer_sizes) - 1
        for l, size_out in enumerate(layer_sizes):
            inputs = fclayer(inputs, size_out, self.wts_init, self.bias_init, self.reuse, self.scope+str(l))

            if l < last:
                inputs = lrelu(inputs)
            else:
                inputs = tf.squeeze(inputs)
        return inputs


    def stack_input(self, decoder_input, lateral_input):
        # Augmented multiplicative term
        mult_input = tf.multiply(decoder_input, lateral_input)
        inputs = tf.stack([decoder_input, lateral_input, mult_input], axis=-1)
        return inputs


class Decoder(object):
    def __init__(self, noisy, clean, scope='dec'):

        self.noisy = noisy
        self.clean = clean
        self.scope = scope
        self.rc_cost = OrderedDict()

        # u_l = tf.expand_dims(tf.cast(self.noisy.predict, tf.float32), axis=-1)  # label, with dim matching
        # self.build(u_l)

    def build(self, decoder_activations, training=True):

        for l in reversed([l for l in self.noisy.z.keys()]):
            # print('dec', l)
            decoder_activations, self.rc_cost[l] = self.compute_rc_cost(l, decoder_activations, training)

    def compute_rc_cost(self, layer, decoder_activations, training=True):
        noisy, clean = self.noisy, self.clean

        # Use decoder weights to upsample the signal from above
        size_out = noisy.layer_sizes[layer]
        u_l = fclayer(decoder_activations, size_out, scope=self.scope + str(layer))

        # Unbatch-normalized activations from parallel layer in noisy encoder
        z_l = noisy.z[layer]

        # Unbatch-normalized target activations from parallel layer in clean encoder
        target_z = clean.z[layer]

        combinator = Combinator(u_l, z_l, layer_sizes=(2, 2, 1), stddev=0.025, scope='com' + str(layer) + '_')
        reconstruction = combinator.outputs

        rc_cost = tf.reduce_sum(
            tf.square(noisy.bn_layers[layer].normalize_from_saved_stats(reconstruction, training=training) - target_z),
            axis=-1)

        return decoder_activations, rc_cost

    # def unsupervised_loss(self, weights):
    #     return sum([self.rc_cost[l] * weights[l] for l in self.rc_cost.keys()])


class GammaDecoder(Decoder):
    def build(self, decoder_activations, training=True):
        l = self.noisy.n_layers-1
        _, self.rc_cost[l] = self.compute_rc_cost(l, decoder_activations, training=training)


def fclayer(input,
            size_out,
            wts_init=layers.xavier_initializer(),
            bias_init=tf.truncated_normal_initializer(stddev=1e-6),
            reuse=None,
            scope=None):
    return layers.fully_connected(
        inputs=input,
        num_outputs=size_out,
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=wts_init,
        weights_regularizer=None,
        biases_initializer=bias_init,
        biases_regularizer=None,
        reuse=reuse,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=scope
    )


def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha*x)


class Ladder(object):

    def __init__(self, params):

        self.params = params
        layer_sizes = params.layer_sizes
        train_flag = params.train_flag
        labeled_batch_size = params.labeled_batch_size
        gamma_flag = params.gamma_flag

        # ===========================
        # ENCODER
        # ===========================

        # PLACEHOLDERS
        # Input placeholder
        self.x = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
        # One-hot targets
        self.y = tf.placeholder(tf.float32, shape=(None, layer_sizes[-1]))

        # CLEAN ENCODER
        self.clean = Encoder(self.x, self.y, layer_sizes, noise_sd=None, reuse=False, training=train_flag)

        # CORRUPTED ENCODER
        self.noisy = Encoder(self.x, self.y, layer_sizes, noise_sd=0.3, reuse=True, training=train_flag)
        self.predict = self.noisy.predict

        # ===========================
        # DECODER
        # ===========================
        if gamma_flag:
            self.decoder = GammaDecoder(self.noisy, self.clean)
        else:
            self.decoder = Decoder(self.noisy, self.clean)

        self.decoder.build(
            decoder_activations=tf.expand_dims(tf.cast(self.predict, tf.float32), axis=-1),
            training=train_flag)

    @property
    def aer(self):
        """
        Compute training error rate on labeled examples only (since e.g. CIFAR-100 with Tiny Images, no labels are actually available)
        At test time, number of labeled examples is same as number of examples"""

        return 1 - tf.reduce_mean(
            tf.cast(tf.equal(self.predict[:self.params.labeled_batch_size], tf.argmax(self.y, 1)), tf.float32))

    @property
    def unsupervised_loss(self):
        return sum([self.decoder.rc_cost[l] * self.params.rc_weights[l] for l in self.decoder.rc_cost.keys()])

    @property
    def training_loss(self):
        return self.loss(train_flag=True)

    @property
    def testing_loss(self):
        return self.loss(train_flag=False)

    def loss(self, train_flag=True):
        # Compute supervised loss on labeled only
        supervised_loss = self.noisy.supervised_loss(
            self.params.labeled_batch_size,
            self.params.unlabeled_batch_size,
            training=train_flag) * self.params.sc_weight

        return self.unsupervised_loss + supervised_loss

        # self.loss = self.supervised_loss + self.unsupervised_loss
        # self.mean_loss = tf.reduce_mean(self.loss)
