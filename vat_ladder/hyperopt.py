import tensorflow as tf
import os
from src.val import build_graph
from src.train import evaluate_metric
from src import input_data
import numpy as np
import argparse
from src.utils import parse_argstring
from skopt import gp_minimize, dump


def get_params(x=None):
    if x is None:
        x = [150, 0.67, 100, 0.002, 0.3, 8.0, 1.0,
             2000, 20, 0.2, 0.2, 0.2, 0.2, 0.2]

    parser = argparse.ArgumentParser()

    # -------------------------
    # Specify at run time of hyperopt
    parser.add_argument('--which_gpu', default=0, type=int)
    parser.add_argument('--num_labeled', default=100, type=int)
    parser.add_argument('--static_bn', default=False, nargs='?', const=0.99, type=float)

    params = parser.parse_args()
    params_dict = vars(params)

    def add(key, default=None, type=None):
        params_dict[key] = default

    # -------------------------
    # Use default values
    add('test_frequency_in_epochs', default=5, type=int)
    add('eval_batch_size', default=100, type=int)
    add('validation', default=1000, type=int)
    add('seed', default=1, type=int)
    add('lr_decay_frequency', default=5, type=int)
    add('batch_size', default=100, type=int)
    add('encoder_layers',
        default=parse_argstring('784-1000-500-250-250-250-10', dtype=int))
    add('num_power_iterations', default=1, type=int)
    add('xi', default=1e-6, type=float)
    add('cnn', False)
    add('ul_batch_size', 100)
    rc_weights = [2000, 20, 0.2, 0.2, 0.2, 0.2, 0.2]
    add('rc_weights', dict(zip(range(len(rc_weights)), rc_weights)))

    # -------------------------
    # Optimize
    add('end_epoch', x[0])
    add('decay_start', x[1])
    add('initial_learning_rate', x[2])
    add('beta1', x[3])
    add('encoder_noise_sd', x[4])
    add('epsilon', x[5])
    add('vat_weight', x[6])
    # add('rc_weights', dict(zip(range(len(x[7:])), x[7:])))

    # Postprocess
    add('decay_start_epoch', int(x[1] * x[0]))

    return params



def func(x=None):

    p = get_params(x)

    # -----------------------------
    # Set GPU device to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.which_gpu)

    # Set seeds
    np.random.seed(p.seed)
    tf.set_random_seed(p.seed)

    # Load data
    print("===  Loading Data ===")
    mnist = input_data.read_data_sets("MNIST_data",
                                      n_labeled=p.num_labeled,
                                      validation_size=p.validation,
                                      one_hot=True,
                                      disjoint=False)
    num_examples = mnist.train.num_examples
    if p.validation > 0:
        mnist.test = mnist.validation
    iter_per_epoch = (num_examples // p.batch_size)
    num_iter = iter_per_epoch * p.end_epoch

    # Build graph
    g, m, trainable_parameters = build_graph(p)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("=== Training ===")
        for i in range(num_iter):
            # if ((i+1) % iter_per_epoch == 0):
            #
            #     error = tf.constant(100.0) - m['acc']
            #     val_err = evaluate_metric(mnist.validation, sess, error,
            #                               graph=g,
            #                               params=p)
            #     print("Epoch {}: {:4.4f}".format(i // iter_per_epoch, val_err))


            images, labels = mnist.train.next_batch(p.batch_size,
                                                    p.ul_batch_size)
            _ = sess.run(
                [g['train_step']],
                feed_dict={g['images']: images,
                           g['labels']: labels,
                           g['train_flag']: True})

        print("=== Evaluating ===")
        error = tf.constant(100.0) - m['acc']
        val_err = evaluate_metric(mnist.validation, sess, error, graph=g,
                                  params=p)

    return val_err


def main():
    dims = [
        (100, 200),                     # 0: end_epoch
        (0.25, 0.99),                   # 1: decay_start
        (0.001, 0.01, 'log-uniform'),   # 2: initial_learning_rate
        (0.5, 0.9),                     # 3: adam beta1
        (0.1, 1.0),                     # 4: encoder_noise_sd
        (0.01, 10.0, 'log-uniform'),    # 5: epsilon
        (0.0, 5.0, 'log-uniform')       # 6: vat_weight
    ]
        # # rc_weights
        # (0.1, 2000, 'log-uniform'), # 0
        # (0.1, 2000, 'log-uniform'), # 1
        # (0.1, 2000, 'log-uniform'), # 2
        # (0.1, 2000, 'log-uniform'), # 3
        # (0.1, 2000, 'log-uniform'), # 4
        # (0.1, 2000, 'log-uniform'), # 5
        # (0.1, 2000, 'log-uniform'), # 6

    x0 = [150, 0.5, 0.002, 0.5, 0.3, 8.0, 1.0]

    res = gp_minimize(func, dims, n_calls=16, x0=x0, verbose=True)
    dump(res, 'hyperopt_res.gz')
    print(res.x, res.fun)



if __name__ == '__main__':
    main()
    # func()
