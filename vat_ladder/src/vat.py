
import tensorflow as tf
from src.ladder import Encoder

# -----------------------------
# VAT FUNCTIONS
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
        self.layer_eps = layer_eps
        self.xi = params.xi
        self.num_power_iters = params.num_power_iters


    def forward(self, x, is_training, update_batch_stats=False):
        # always use a standard Gaussian-noise encoder
        vatfw = Encoder(
            inputs=x,
            bn=self.bn,
            is_training=is_training,
            params=self.params,  # for encoder_layers, batch_size, van settings
            this_encoder_noise=self.params.vadv_sd, # add gaussian for stability
            start_layer=self.start_layer,
            update_batch_stats=update_batch_stats,
            scope='enc', reuse=True)
        return vatfw.logits  # logits by default includes both labeled/unlabeled

    def generate_virtual_adversarial_perturbation(self, x, logit, is_training):
        d = tf.random_normal(shape=tf.shape(x))
        for k in range(self.num_power_iters):
            d = self.xi * get_normalized_vector(d)
            logit_p = logit
            print("=== Power Iteration: {} ===".format(k))
            logit_m = self.forward(x + d, update_batch_stats=False,
                              is_training=is_training)
            dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
        return self.layer_eps * get_normalized_vector(d)

    def generate_adversarial_perturbation(self, x, loss):
        grad = tf.gradients(loss, [x], aggregation_method=2)[0]
        grad = tf.stop_gradient(grad)
        return self.layer_eps * get_normalized_vector(grad)

    def adversarial_loss(self, x, y, loss, is_training,
                         name="at_loss"):
        r_adv = self.generate_adversarial_perturbation(x, loss)
        logit = self.forward(x + r_adv, is_training=is_training,
                        update_batch_stats=False)
        loss = ce_loss(logit, y)
        return tf.identity(loss, name=name)

    def virtual_adversarial_loss(self, x, logit, is_training,
                                 name="vat_loss"):

        print("=== VAT Pass: Generating VAT perturbation ===")
        r_vadv = self.generate_virtual_adversarial_perturbation(
            x, logit, is_training=is_training)
        logit = tf.stop_gradient(logit)
        logit_p = logit

        print("=== VAT Pass: Computing VAT Loss (KL Divergence)")
        logit_m = self.forward(x + r_vadv, update_batch_stats=False,
                          is_training=is_training)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return tf.identity(loss, name=name)
