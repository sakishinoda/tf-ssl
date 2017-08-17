
import tensorflow as tf

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

def get_normalized_vector(d):

    d_dims = len(d.get_shape()) - 1
    axes = [range(1, d_dims)] if d_dims > 1 else [1]
    d /= (1e-12 + tf.reduce_max(tf.abs(d), axis=axes, keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), axis=axes,
                                      keep_dims=True))
    return d

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

class Adversary(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def forward(self, x, is_training,
                update_batch_stats=False,
                start_layer=0):
        """

        :param x: input at start_layer
        :param is_training: tensorflow bool
        :param update_batch_stats: python bool
        :param start_layer:
        :return:
        """
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
