
# import IPython
import argparse
import tensorflow as tf
# import vat_mlp as vat
import layers as L
import numpy as np
import os
from mnist import read_data_sets
import time
import math


parser = argparse.ArgumentParser()

parser.add_argument('--id', default='VAT')
parser.add_argument('--which_gpu', default='0')
parser.add_argument('--log_dir', default='')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--ul_batch_size', default=250, type=int)
parser.add_argument('--eval_batch_size', default=100, type=int)
parser.add_argument('--eval_freq', default=5, type=int)
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--method', default='vat') # 'vat', 'vatent', 'baseline'
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--num_labeled', default=100, type=int)
parser.add_argument('--epsilon', default=0.3, type=float) # 0.3 for SSL MNIST
parser.add_argument('--num_power_iterations', default=1, type=int)
parser.add_argument('--xi', default=1e-6, type=float)
parser.add_argument('--epoch_decay_start', default=80, type=int)
parser.add_argument('--mom1', default=0.9, type=float)
# parser.add_argument('--mom2', default=0.5, type=float)
params = parser.parse_args()

params.num_iter_per_epoch = 240
params.lrelu_a = 0.1
params.top_bn = False

def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):

    def weight(s, i):
        return tf.get_variable('w'+str(i), shape=s,
                               initializer=tf.random_normal_initializer(stddev=(
                               1/math.sqrt(sum(s)))))
    def bias(s, i):
        return tf.get_variable('b'+str(i), shape=s,
                               initializer=tf.zeros_initializer)

    h = L.lrelu(tf.matmul(x, weight([784, 1200], 1)) + bias([1200], 1))

    h = L.bn(h, 1200, is_training=is_training,
             update_batch_stats=update_batch_stats, name='bn1')

    h = L.lrelu(tf.matmul(h, weight([1200, 1200], 2)) + bias([1200], 2))

    h = L.bn(h, 1200, is_training=is_training,
             update_batch_stats=update_batch_stats, name='bn2')

    h = tf.matmul(h, weight([1200, 10], 3)) + bias([10], 3)

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

    for _ in range(params.num_power_iterations):
        d = params.xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return params.epsilon * get_normalized_vector(d)


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
    return params.epsilon * get_normalized_vector(grad)


def adversarial_loss(x, y, loss, is_training=True, name="at_loss"):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = forward(x + r_adv, is_training=is_training, update_batch_stats=False)
    loss = L.ce_loss(logit, y)
    return loss



def build_training_graph(x, y, ul_x, lr):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    logit = forward(x)

    nll_loss = L.ce_loss(logit, y)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        ul_logit = forward(ul_x, is_training=True,
                           update_batch_stats=False)
        vat_loss = virtual_adversarial_loss(ul_x, ul_logit)
        loss = nll_loss + vat_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def accuracy(x, y):
    logit = forward(x, is_training=False, update_batch_stats=False)
    pred = tf.argmax(logit, 1)
    true = tf.argmax(y, 1)
    return tf.reduce_mean(tf.to_float(tf.equal(pred, true)))


def main():
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.which_gpu)

    mnist = read_data_sets('MNIST_data', one_hot=True,
                           n_labeled=params.num_labeled,
                           disjoint=False)

    # Training
    inputs = tf.placeholder(tf.float32, shape=(None, 784))
    outputs = tf.placeholder(tf.float32, shape=(None, 10))
    ul_inputs = tf.placeholder(tf.float32, shape=(params.ul_batch_size, 784))

    lr = tf.placeholder_with_default(params.learning_rate, shape=[], name="learning_rate")
    # mom = tf.placeholder_with_default(params.mom1, shape=[], name="momentum")
    # train_flag = tf.placeholder(tf.bool)


    with tf.variable_scope('MLP', reuse=None) as scope:
        loss, train_op, global_step = build_training_graph(inputs, outputs,
                                                           ul_inputs, lr)
        scope.reuse_variables()
        acc_op = accuracy(inputs, outputs)

    init_op = tf.global_variables_initializer()

    # -----------------------------
    # Write logs to appropriate directory
    log_dir = "logs/" + params.id
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/" + "train_log"
    acc_file = log_dir + "/" + "acc_log"


    with tf.Session() as sess:
        sess.run(init_op)
        for ep in range(params.num_epochs):
            if ep < params.epoch_decay_start:
                # feed_dict = {lr: params.learning_rate, mom: params.mom1}
                feed_dict = {lr: params.learning_rate}
            else:
                decayed_lr = ((params.num_epochs - ep) / float(
                    params.num_epochs - params.epoch_decay_start)) * params.learning_rate
                # feed_dict = {lr: decayed_lr, mom: params.mom2}
                feed_dict = {lr: decayed_lr}

            sum_loss = 0
            start = time.time()

            # TRAINING
            for i in range(params.num_iter_per_epoch):
                images, labels, ul_images = mnist.train.next_batch(params.batch_size, params.ul_batch_size)
                feed_dict[inputs] = images
                feed_dict[outputs] = labels
                feed_dict[ul_inputs] = ul_images
                _, batch_loss, _ = sess.run([train_op, loss, global_step],
                                            feed_dict=feed_dict)
                sum_loss += batch_loss

            end = time.time()
            with open(log_file, 'a') as train_log:
                print("Epoch:", ep, "CE_loss_train:",
                      sum_loss / params.num_iter_per_epoch, "elapsed_time:",
                      end - start, file=train_log, flush=True)

            # EVAL
            if (ep + 1) % params.eval_freq == 0 or ep + 1 == params.num_epochs:

                eval_acc = dict(train = 0, test = 0)

                # Eval on training data
                n_iter_per_epoch = mnist.train_eval.num_examples // \
                                   params.eval_batch_size

                for i in range(n_iter_per_epoch):

                    test_images, test_labels = \
                        mnist.train.next_batch(params.eval_batch_size)

                    eval_feed_dict = {
                        inputs: test_images,
                        outputs: test_labels
                    }

                    acc_val = sess.run(acc_op, eval_feed_dict)

                    eval_acc['train'] += acc_val

                eval_acc['train'] /= n_iter_per_epoch
                
                # Eval on testing data

                n_iter_per_epoch = mnist.test.num_examples // \
                                   params.eval_batch_size

                for i in range(n_iter_per_epoch):

                    test_images, test_labels = \
                        mnist.test.next_batch(params.eval_batch_size)

                    eval_feed_dict = {
                        inputs: test_images,
                        outputs: test_labels
                    }

                    acc_val = sess.run(acc_op, eval_feed_dict)

                    eval_acc['test'] += acc_val

                eval_acc['test'] /= n_iter_per_epoch
                
                with open(acc_file, 'a') as acc_log:
                    print(ep, eval_acc['train'], eval_acc['test'], file=acc_log)


if __name__ == "__main__" :
    main()
