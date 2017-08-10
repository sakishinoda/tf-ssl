import tensorflow as tf
import numpy
import sys, os
import math

import layers as L
# import cnn
import tensorflow.contrib.layers as tfl


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('epsilon', 8.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('xi', 1e-6, "small constant for finite difference")

# FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.app.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.app.flags.DEFINE_boolean('top_bn', False, "")


def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):

    def weight(s, i):
        return tf.get_variable('w'+str(i), shape=s,
                               initializer=tf.random_normal_initializer(stddev=(
                               1/math.sqrt(sum(s)))))
    def bias(s, i):
        return tf.get_variable('b'+str(i), shape=s,
                               initializer=tf.zeros_initializer)

    h = tf.nn.relu(tf.matmul(x, weight([784, 1200], 1))+bias([1200], 1))

    h = L.bn(h, 1200, is_training=is_training,
             update_batch_stats=update_batch_stats, name='bn1')

    h = tf.nn.relu(tf.matmul(h, weight([1200, 1200], 2))+bias([1200], 2))

    h = L.bn(h, 1200, is_training=is_training,
             update_batch_stats=update_batch_stats, name='bn2')

    h = tf.matmul(h, weight([1200, 10], 3), + bias([10], 3))

    return h
    # return cnn.logit(x, is_training=is_training,
    #                  update_batch_stats=update_batch_stats,
    #                  stochastic=stochastic,
    #                  seed=seed)


def forward(x, is_training=True, update_batch_stats=True, seed=1234):
    if is_training:
        return logit(x, is_training=True,
                     update_batch_stats=update_batch_stats,
                     stochastic=True, seed=seed)
    else:
        return logit(x, is_training=False,
                     update_batch_stats=update_batch_stats,
                     stochastic=False, seed=seed)


def get_normalized_vector(d):
    red_axes = list(range(1, len(d.get_shape())))
    # print(d.get_shape(), red_axes)
    d /= (1e-12 + tf.reduce_max(tf.abs(d), axis=red_axes,
                                            keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0),
                                      axis=red_axes,
                                      keep_dims=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, is_training=True):
    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(FLAGS.num_power_iterations):
        d = FLAGS.xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return FLAGS.epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, is_training=True, name="vat_loss"):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, is_training=is_training)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward(x + r_vadv, update_batch_stats=False, is_training=is_training)
    loss = L.kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)


def generate_adversarial_perturbation(x, loss):
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return FLAGS.epsilon * get_normalized_vector(grad)


def adversarial_loss(x, y, loss, is_training=True, name="at_loss"):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = forward(x + r_adv, is_training=is_training, update_batch_stats=False)
    loss = L.ce_loss(logit, y)
    return loss
