import argparse
import tensorflow as tf
import vat_mlp as vat
import layers as L
import numpy as np
import os
from mnist import read_data_sets

parser = argparse.ArgumentParser()

parser.add_argument('--device', default='0')
parser.add_argument('--log_dir', default='')
parser.add_argument('--seed', default=1)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--ul_batch_size', default=128)
parser.add_argument('--eval_batch_size', default=100)
parser.add_argument('--eval-freq', default=5)
parser.add_argument('--num_epochs', default=120)
parser.add_argument('--method', default='vat', choose=['vat', 'vatent',
                                                       'baseline'])

FLAGS = parser.parse_args()

FLAGS.epochs_decay_start = 80
FLAGS.num_iter_per_epoch = 400
FLAGS.learning_rate = 0.001
FLAGS.mom1 = 0.9
FLAGS.mom2 = 0.5





def build_training_graph(x, y, ul_x, lr, mom):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    logit = vat.forward(x)
    nll_loss = L.ce_loss(logit, y)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        if FLAGS.method == 'vat':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            additional_loss = vat_loss
        elif FLAGS.method == 'vatent':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            ent_loss = L.entropy_y_x(ul_logit)
            additional_loss = vat_loss + ent_loss
        elif FLAGS.method == 'baseline':
            additional_loss = 0
        else:
            raise NotImplementedError
        loss = nll_loss + additional_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step



def build_eval_graph(x, y, ul_x):
    losses = {}
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    at_loss = vat.adversarial_loss(x, y, nll_loss, is_training=True)
    losses['AT_loss'] = at_loss
    ul_logit = vat.forward(ul_x, is_training=False, update_batch_stats=False)
    vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit, is_training=False)
    losses['VAT_loss'] = vat_loss
    return losses


def main():
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.which_gpu)


    mnist = read_data_sets('MNIST_data', one_hot=True, n_labeled=100)

    inputs = tf.placeholder(tf.float32, shape=(None, ls[0]))
    outputs = tf.placeholder(tf.float32)
