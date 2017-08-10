
import IPython
import argparse
import tensorflow as tf
import vat_mlp as vat
import layers as L
import numpy as np
import os
from mnist import read_data_sets
import time


parser = argparse.ArgumentParser()

parser.add_argument('--id', default='VAT')
parser.add_argument('--which_gpu', default='0')
parser.add_argument('--log_dir', default='')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--ul_batch_size', default=128)
parser.add_argument('--eval_batch_size', default=100)
parser.add_argument('--eval_freq', default=5)
parser.add_argument('--num_epochs', default=120)
parser.add_argument('--method', default='vat') # 'vat', 'vatent', 'baseline'

params = parser.parse_args()


params.epoch_decay_start = 80
params.num_iter_per_epoch = 400
params.learning_rate = 0.001
params.mom1 = 0.9
params.mom2 = 0.5

# IPython.embed()

def build_training_graph(x, y, ul_x, lr, mom):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    logit = vat.forward(x)
    print(logit.get_shape(), y.get_shape())
    # IPython.embed()
    nll_loss = L.ce_loss(logit, y)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        if params.method == 'vat':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            additional_loss = vat_loss
        elif params.method == 'vatent':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            ent_loss = L.entropy_y_x(ul_logit)
            additional_loss = vat_loss + ent_loss
        elif params.method == 'baseline':
            additional_loss = 0
        else:
            raise NotImplementedError
        loss = nll_loss + additional_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def accuracy(x, y):
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    return L.accuracy(logit, y)


def main():
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.which_gpu)

    mnist = read_data_sets('MNIST_data', one_hot=True, n_labeled=100)

    # Training
    training_placeholders = dict(
        inputs = tf.placeholder(tf.float32, shape=(params.batch_size, 784)),
        outputs = tf.placeholder(tf.float32, shape=(params.batch_size, 10)),
        ul_inputs = tf.placeholder(tf.float32, shape=(params.ul_batch_size, 784))

    )
    eval_placeholders = dict(
        inputs=tf.placeholder(tf.float32, shape=(params.eval_batch_size, 784)),
        outputs=tf.placeholder(tf.float32, shape=(params.eval_batch_size, 10))
    )

    lr = tf.placeholder_with_default(params.learning_rate, shape=[], name="learning_rate")
    mom = tf.placeholder_with_default(params.mom1, shape=[], name="momentum")

    with tf.variable_scope("MLP", reuse=None) as scope:
        loss, train_op, global_step = build_training_graph(
            training_placeholders['inputs'],
            training_placeholders['outputs'],
            training_placeholders['ul_inputs'],
            lr, mom)

        scope.reuse_variables()
        acc_op = accuracy(
            eval_placeholders['inputs'],
            eval_placeholders['outputs'])

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
                feed_dict = {lr: params.learning_rate, mom: params.mom1}
            else:
                decayed_lr = ((params.num_epochs - ep) / float(
                    params.num_epochs - params.epoch_decay_start)) * params.learning_rate
                feed_dict = {lr: decayed_lr, mom: params.mom2}

            sum_loss = 0
            start = time.time()

            # TRAINING
            for i in range(params.num_iter_per_epoch):
                images, labels, ul_images = mnist.train.next_batch(params.batch_size, params.ul_batch_size)
                feed_dict[training_placeholders['inputs']] = images
                feed_dict[training_placeholders['outputs']] = labels
                feed_dict[training_placeholders['ul_inputs']] = ul_images
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
                        mnist.test.next_batch(params.eval_batch_size)

                    eval_feed_dict = {
                        eval_placeholders['inputs']: test_images,
                        eval_placeholders['outputs']: test_labels
                    }

                    acc_val = sess.run(acc_op, eval_feed_dict)

                    eval_acc['train'] += acc_val


                # Eval on testing data

                n_iter_per_epoch = mnist.test.num_examples // \
                                   params.eval_batch_size

                for i in range(n_iter_per_epoch):

                    test_images, test_labels = \
                        mnist.test.next_batch(params.eval_batch_size)

                    eval_feed_dict = {
                        eval_placeholders['inputs']: test_images,
                        eval_placeholders['outputs']: test_labels
                    }

                    acc_val = sess.run(acc_op, eval_feed_dict)

                    eval_acc['test'] += acc_val

                with open(acc_file, 'a') as acc_log:
                    print(ep, eval_acc['train'], eval_acc['test'], file=acc_log)


if __name__ == "__main__" :
    main()
