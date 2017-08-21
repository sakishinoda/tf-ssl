
import tensorflow as tf
import os
import time
from tqdm import tqdm
from src.utils import get_cli_params, process_cli_params, \
    order_param_settings
from src.val import build_graph, measure_smoothness
from src.train import evaluate_metric_list, update_decays
from src import input_data
import numpy as np


def main():

    p = process_cli_params(get_cli_params())

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
    p.num_examples = num_examples
    if p.validation > 0:
        mnist.test = mnist.validation
    p.iter_per_epoch = (num_examples // p.batch_size)
    p.num_iter = p.iter_per_epoch * p.end_epoch

    # -----------------------------
    # Build graph
    g, m, trainable_parameters = build_graph(p)

    # Collect losses
    train_losses = [m['loss'], m['cost'], m['uc'], m['vc']]
    test_losses = [m['cost']]

    if p.measure_smoothness:
        s = measure_smoothness(g, p)
    #     print(s.get_shape())
        train_losses.append(tf.reduce_mean(s))

    # Write logs to appropriate directory
    log_dir = p.logdir + p.id
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    desc_file = log_dir + "/" + "description"
    with open(desc_file, 'a') as f:
        print(*order_param_settings(p), sep='\n', file=f, flush=True)
        print("Trainable parameters:", trainable_parameters, file=f,
              flush=True)

    log_file = log_dir + "/" + "train_log"

    # -----------------------------
    print("===  Starting Session ===")
    sess = tf.Session()
    i_iter = 0
    # -----------------------------
    # Resume from checkpoint
    ckpt_dir = "checkpoints/" + p.id + "/"
    ckpt = tf.train.get_checkpoint_state(
        ckpt_dir)  # get latest checkpoint (if any)

    if ckpt and ckpt.model_checkpoint_path:
        # if checkpoint exists,
        # restore the parameters
        # and set epoch_n and i_iter
        g['saver'].restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
        i_iter = (epoch_n + 1) * (p.num_examples // p.batch_size)
        print("Restored Epoch ", epoch_n)

    else:
        # no checkpoint exists.
        # create checkpoints directory if it does not exist.
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        init = tf.global_variables_initializer()
        sess.run(init)

    # -----------------------------
    print("=== Training ===")
    # -----------------------------
    eval_metrics = lambda dataset, sess, ops: evaluate_metric_list(
        dataset, sess, ops, graph=g, params=p)

    # Evaluate initial training accuracy and losses
    # init_loss = evaluate_metric(
    # mnist.train.labeled_ds, sess, cost)
    with open(desc_file, 'a') as f:
        print('================================', file=f, flush=True)
        print("Initial Train Accuracy: ",
              sess.run(m['acc'], feed_dict={
                  g['images']: mnist.train.labeled_ds.images,
                  g['labels']: mnist.train.labeled_ds.labels,
                  g['train_flag']: False}),
              "%", file=f, flush=True)
        # print("Initial Train Losses: ", *eval_metrics(
        #     mnist.train, sess, train_losses), file=f,
        #       flush=True)

        # -----------------------------
        # Evaluate initial testing accuracy and cross-entropy loss
        print("Initial Test Accuracy: ",
              sess.run(m['acc'], feed_dict={
                  g['images']: mnist.test.images,
                  g['labels']: mnist.test.labels,
                  g['train_flag']: False}),
              "%", file=f, flush=True)
        # print("Initial Test Losses: ",
        #       *eval_metrics(
        #           mnist.test, sess, test_losses), file=f,
        #       flush=True)

    start = time.time()
    for i in tqdm(range(i_iter, p.num_iter)):

        images, labels = mnist.train.next_batch(p.batch_size, p.ul_batch_size)

        _ = sess.run(
            [g['train_step']],
            feed_dict={g['images']: images,
                       g['labels']: labels,
                       g['train_flag']: True})

        epoch_n = i // (p.num_examples // p.batch_size)
        # ---------------------------------------------
        # Epoch completed?
        if (i > 1) and ((i + 1) % p.iter_per_epoch == 0):
            update_decays(sess, epoch_n, iter=i, graph=g, params=p)

        # ---------------------------------------------
        # Evaluate every test_frequency_in_epochs
        if (i > 1) and ((i + 1) % int(p.test_frequency_in_epochs *
                           p.iter_per_epoch) == 0):

            now = time.time() - start

            if not p.do_not_save:
                g['saver'].save(sess, ckpt_dir + 'model.ckpt', epoch_n)

            # ---------------------------------------------
            # Compute error on testing set (10k examples)
            test_costs = eval_metrics(mnist.test, sess,
                                      test_losses)
            train_costs = sess.run(
                train_losses,
                feed_dict={g['images']: images,
                           g['labels']: labels,
                           g['train_flag']: False})

            # Create log of:
            # time, epoch number, test accuracy, test cross entropy,
            # train accuracy, train loss, train cross entropy,
            # train reconstruction loss, smoothness

            log_i = [now, epoch_n] + sess.run(
                [m['acc']],
                feed_dict={g['images']: mnist.test.images,
                           g['labels']: mnist.test.labels,
                           g['train_flag']: False}
            ) + test_costs + sess.run(
                [m['acc']],
                feed_dict={g['images']:
                               mnist.train.labeled_ds.images,
                           g['labels']: mnist.train.labeled_ds.labels,
                           g['train_flag']: False}
            ) + train_costs

            with open(log_file, 'a') as train_log:
                print(*log_i, sep=',', flush=True, file=train_log)


    with open(desc_file, 'a') as f:
        print("Final Accuracy: ", sess.run(m['acc'], feed_dict={
            g['images']: mnist.test.images, g['labels']:
                mnist.test.labels,
            g['train_flag']: False}),
              "%", file=f, flush=True)

    sess.close()


if __name__ == '__main__':
    main()








