
import tensorflow as tf
import tensorflow.contrib.layers as layers
import math

def get_batch_ops(batch_size):
    join = lambda l, u: tf.concat([l, u], 0)
    split_lu = lambda x: (labeled(x), unlabeled(x))
    labeled = lambda x: x[:batch_size] if x is not None else x
    unlabeled = lambda x: x[batch_size:] if x is not None else x
    return join, split_lu, labeled, unlabeled

def fclayer(input, size_out, wts_init=layers.xavier_initializer(), bias_init=tf.constant_initializer(1e-6), reuse=None, scope=None,
            activation=None):
    return layers.fully_connected(
        inputs=input,
        num_outputs=size_out,
        activation_fn=activation,
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


class Weights(object):
    def __init__(self, params):

        def bias_init(inits, size, name, scope):
            with tf.variable_scope(scope, reuse=None):
                return tf.get_variable(name, initializer=inits)

        def wts_init(shape, name, scope):
            with tf.variable_scope(scope, reuse=None):
                return tf.get_variable(name, tf.random_normal(shape, stddev =
            math.sqrt(shape[0])))

        if params.cnn:
            LS = params.cnn_fan
            weights = {}
        else:
            LS = params.encoder_layers
            shapes = list(zip(LS[:-1], LS[1:]))

            # # Encoder weights
            # self.W = [wts_init(s, name=str(i), scope='W') for i, s in
            #           enumerate(shapes)]
            # # Decoder weights
            # self.V = [wts_init(s[::-1], name=str(i), scope='V') for i, s in
            #           enumerate(shapes)]

        # batch normalization parameter to shift the normalized value
        self.beta = [bias_init(0.0, LS[l + 1], name=str(l), scope="beta")
                     for l in range(params.num_layers)]
        # batch normalization parameter to scale the normalized value
        self.gamma = [bias_init(1.0, LS[l + 1], name=str(l), scope="gamma")
                      for l in range(params.num_layers)]


class Activations(object):
    """Store statistics for each layer in the encoder/decoder structures."""
    def __init__(self):
        self.z = {} # pre-activation
        self.h = {} # activation
        self.m = {} # mean
        self.v = {} # variance


class Encoder(object):
    def __init__(self,
                 params,
                 inputs,
                 start_layer,
                 bn,
                 is_training,
                 weights,
                 noise_std = 0.0,
                 update_batch_stats=True):
        self.params = params
        self.noise_std = noise_std
        self.labeled = Activations()
        self.unlabeled = Activations()
        join, split_lu, labeled, unlabeled = get_batch_ops(params.batch_size)

        ls = params.encoder_layers  # seq of layer sizes, len num_layers


        # Layer 0: inputs, size 784
        l = 0
        h = inputs + self.generate_noise(inputs, l)
        self.labeled.z[l], self.unlabeled.z[l] = split_lu(h)

        for l in range(start_layer, params.num_layers + 1):
            print("Layer {}: {} -> {}".format(l, ls[l - 1], ls[l]))
            self.labeled.h[l-1], self.unlabeled.z[l-1] = split_lu(h)
            # z_pre = tf.matmul(h, self.W[l-1])
            z_pre = layers.fully_connected(h, num_outputs=ls[l])
            z_pre_l, z_pre_u = split_lu(z_pre)
            m, v = tf.nn.moments(z_pre_u, axes=[0])
            # save mean and variance of unlabeled examples for decoding
            self.unlabeled.m[l], self.unlabeled.v[l] = m, v

            # if training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                # if noise_std > 0:
                if not update_batch_stats:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = join(bn.batch_normalization(z_pre_l),
                             bn.batch_normalization(z_pre_u, m, v))
                    noise = self.generate_noise(z_pre, l)
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
                mean = bn.ewma.average(bn.running_mean[l - 1])
                var = bn.ewma.average(bn.running_var[l - 1])
                z = bn.batch_normalization(z_pre, mean, var)

                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(is_training, training_batch_norm, eval_batch_norm)

            if l == params.num_layers:
                # return pre-softmax logits in final layer
                logits = weights.gamma[l - 1] * (z + weights.beta[l - 1])
                h = tf.nn.softmax(logits)
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + weights.beta[l - 1])

            self.labeled.z[l], self.labeled.z[l] = split_lu(z)
            self.labeled.h[l], self.unlabeled.h[l] = split_lu(h)

        self.logits = logits


    def generate_noise(self, inputs, l):
        """Add noise depending on corruption parameters"""
        # start_layer = l+1
        # corrupt = self.params.corrupt
        # if corrupt == 'vatgauss':
        #     noise = generate_virtual_adversarial_perturbation(
        #         inputs, clean_logits, is_training=is_training,
        #         start_layer=start_layer) + \
        #         tf.random_normal(tf.shape(inputs)) * noise_std
        # elif corrupt == 'vat':
        #     noise = generate_virtual_adversarial_perturbation(
        #         inputs, clean_logits, is_training=is_training,
        #         start_layer=start_layer)
        # elif corrupt == 'gauss':
        #     noise = tf.random_normal(tf.shape(inputs)) * noise_std
        # else:
        #     noise = tf.zeros(tf.shape(inputs))
        # return noise
        return tf.random_normal(tf.shape(inputs)) * self.noise_std


class Decoder(object):

    def __init__(self, params, clean, corr, weights, bn, combinator):
        """

        :param params: namespace
        :param clean: clean encoder object
        :param corr: corrupted encoder object
        """
        self.params = params
        ls = params.encoder_layers  # seq of layer sizes, len num_layers
        denoising_cost = params.rc_weights
        join, split_lu, labeled, unlabeled = get_batch_ops(params.batch_size)

        z_est = {}  # activation reconstruction
        d_cost = []  # denoising cost

        for l in range(params.num_layers, -1, -1):
            print("Layer {}: {} -> {}, denoising cost: {}".format(
                l, ls[l + 1] if l + 1 < len(ls) else None,
                ls[l], denoising_cost[l]
            ))

            z, z_c = clean.unlabeled.z[l], corr.unlabeled.z[l]
            m, v = clean.unlabeled.m.get(l, 0), \
                   clean.unlabeled.v.get(l, 1 - 1e-10)
            # print(l)
            if l == params.num_layers:
                u = unlabeled(corr.logits)
            else:
                u = tf.matmul(z_est[l + 1], weights.V[l])

            u = bn.batch_normalization(u)

            z_est[l] = combinator(z_c, u, ls[l])

            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append((tf.reduce_mean(
                tf.reduce_sum(tf.square(z_est_bn - z), 1)) / ls[l]) *
                          denoising_cost[l])

        self.z_est = z_est
        self.d_cost = d_cost



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


def preprocess(placeholder, params):
    return tf.reshape(placeholder, shape=[
        -1, params.cnn_init_size, params.cnn_init_size, params.cnn_fan[0]
    ]) if params.cnn else placeholder


def main(params):
    ls = params.cnn_fan if params.cnn else params.encoder_layers
    images_placeholder = tf.placeholder(tf.float32, shape=(None, ls[0]))
    images = preprocess(images_placeholder, params)
    labels = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    weights = Weights(params)

    bn = BatchNormLayers(ls, params)
    print("=== Clean Encoder ===")
    with tf.variable_scope('enc', reuse=None):
        clean = Encoder(params, images, 0, bn, train_flag, weights, 0.0, True)

    print("=== Corrupted Encoder === ")
    with tf.variable_scope('enc', reuse=True):
        corr = Encoder(params, images, 0, bn, train_flag, weights,
                       params.encoder_noise_std, False)














