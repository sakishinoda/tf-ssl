
import tensorflow as tf
import os
import time
from tqdm import tqdm
from src.utils import get_cli_params, process_cli_params, \
    order_param_settings, count_trainable_params, preprocess, get_batch_ops
from src.vat import Adversary
from src.ladder import Ladder
from src import input_data
import numpy as np


def main():

    params = process_cli_params(get_cli_params())

    # -----------------------------
    # Set GPU device to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.which_gpu)

    # Set seeds
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    print("===  Loading Data ===")
    mnist = input_data.read_data_sets("MNIST_data",
                                      n_labeled=params.num_labeled,
                                      one_hot=True,
                                      disjoint=False)
    num_examples = mnist.train.num_examples
    # -----------------------------
    # Parameter setup
    params.iter_per_epoch = (num_examples // params.batch_size)
    params.num_iter = params.iter_per_epoch * params.end_epoch
    params.encoder_layers = params.cnn_fan if params.cnn else \
        params.encoder_layers

    # -----------------------------
    # Placeholder setup
    inputs_placeholder = tf.placeholder(
        tf.float32, shape=(None, params.encoder_layers[0]))
    inputs = preprocess(inputs_placeholder, params)
    outputs = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # -----------------------------
    # Ladder
    ladder = Ladder(inputs, outputs, train_flag, params)

    # -----------------------------
    # Add top-level VAT/AT cost
    vadv = Adversary(ladder.bn, params)
    join, split_lu, labeled, unlabeled = get_batch_ops(params.batch_size)

    # AT on labeled only
    # at_cost = adversarial_loss(x=labeled(inputs),
    #                            y=outputs,
    #                            loss=ladder.cost,
    #                            is_training=train_flag,
    #                            start_layer=0
    #                            ) * params.at_weight
    # at_cost = tf.zeros(shape=tf.shape(labeled(inputs)))

    # VAT on unlabeled only
    vat_cost = vadv.virtual_adversarial_loss(
        x=unlabeled(inputs),
        logit=unlabeled(ladder.corr.logits),
        is_training=train_flag,
        start_layer=0) * params.vat_weight

    # -----------------------------
    # Loss, accuracy and training steps
    loss = ladder.cost + ladder.u_cost + vat_cost
    train_losses = [loss, ladder.cost, ladder.u_cost, vat_cost]
    test_losses = [ladder.cost]

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(ladder.predict, tf.argmax(outputs, 1)),
            "float")) * tf.constant(100.0)

    learning_rate = tf.Variable(params.initial_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*ladder.bn.bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=5)

    # -----------------------------
    # Create logs after full graph created to count trainable parameters
    # Write logs to appropriate directory
    log_dir = params.logdir + params.id
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    desc_file = log_dir + "/" + "description"
    with open(desc_file, 'a') as f:
        print(*order_param_settings(params), sep='\n', file=f, flush=True)
        print("Trainable parameters:", count_trainable_params(), file=f,
              flush=True)

    log_file = log_dir + "/" + "train_log"

    # -----------------------------
    print("===  Starting Session ===")
    sess = tf.Session()
    i_iter = 0
    # -----------------------------
    # Resume from checkpoint
    ckpt_dir = "checkpoints/" + params.id + "/"
    ckpt = tf.train.get_checkpoint_state(
        ckpt_dir)  # get latest checkpoint (if any)
    if ckpt and ckpt.model_checkpoint_path:
        # if checkpoint exists, restore the parameters and set epoch_n and i_iter
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
        i_iter = (epoch_n + 1) * (num_examples // params.batch_size)
        print("Restored Epoch ", epoch_n)
    else:
        # no checkpoint exists. create checkpoints directory if it does not exist.
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        init = tf.global_variables_initializer()
        sess.run(init)


    # -----------------------------
    print("=== Training ===")

    def evaluate_metric(dataset, sess, op):
        metric = 0
        num_eval_iters = dataset.num_examples // params.batch_size
        for _ in range(num_eval_iters):
            images, labels = dataset.next_batch(params.batch_size)
            init_feed = {inputs_placeholder: images,
                         outputs: labels,
                         train_flag: False}
            metric += sess.run(op, init_feed)
        metric /= num_eval_iters
        return metric

    def evaluate_metric_list(dataset, sess, ops):
        metrics = [0.0 for _ in ops]
        num_eval_iters = dataset.num_examples // params.batch_size
        for _ in range(num_eval_iters):
            images, labels = dataset.next_batch(params.batch_size)
            init_feed = {inputs_placeholder: images,
                         outputs: labels,
                         train_flag: False}
            op_eval = sess.run(ops, init_feed)

            for i, op in enumerate(op_eval):
                metrics[i] += op

        metrics = [metric/num_eval_iters for metric in metrics]
        return metrics

    # -----------------------------
    # Evaluate initial training accuracy and losses
    # init_loss = evaluate_metric(
        # mnist.train.labeled_ds, sess, cost)
    with open(desc_file, 'a') as f:
        print('================================', file=f, flush=True)
        print("Initial Train Accuracy: ",
              sess.run(accuracy, feed_dict={
                  inputs_placeholder: mnist.train.labeled_ds.images,
                  outputs: mnist.train.labeled_ds.labels,
                  train_flag: False}),
              "%", file=f, flush=True)
        print("Initial Train Losses: ", *evaluate_metric_list(
            mnist.train, sess, train_losses), file=f,
              flush=True)

        # -----------------------------
        # Evaluate initial testing accuracy and cross-entropy loss
        print("Initial Test Accuracy: ",
              sess.run(accuracy, feed_dict={
                  inputs_placeholder: mnist.test.images,
                  outputs: mnist.test.labels,
                  train_flag: False}),
              "%", file=f, flush=True)
        print("Initial Test Losses: ",
              *evaluate_metric_list(
                  mnist.test, sess, test_losses), file=f,
              flush=True)

    start = time.time()
    for i in tqdm(range(i_iter, params.num_iter)):

        images, labels = mnist.train.next_batch(params.batch_size)

        _ = sess.run(
            [train_step],
            feed_dict={inputs_placeholder: images,
                       outputs: labels,
                       train_flag: True})

        # ---------------------------------------------
        # Epoch completed?
        if (i > 1) and ((i+1) % params.iter_per_epoch == 0):
            epoch_n = i // (num_examples // params.batch_size)

            # ---------------------------------------------
            # Update batch norm decay constant
            if params.bn_decay == 'dynamic':
                ladder.bn_decay.assign(1.0 - (1.0 / (epoch_n + 1)))

            # ---------------------------------------------
            # Update learning rate every epoch
            if (epoch_n + 1) >= params.decay_start_epoch:
                # epoch_n + 1 because learning rate is set for next epoch
                ratio = 1.0 * (params.end_epoch - (epoch_n + 1))
                ratio = max(0., ratio / (params.end_epoch - params.decay_start_epoch))
                sess.run(learning_rate.assign(params.initial_learning_rate *
                                              ratio))

            # ---------------------------------------------
            # Evaluate every test_frequency_in_epochs
            if ((i + 1) % (params.test_frequency_in_epochs *
                               params.iter_per_epoch) == 0):
                now = time.time() - start

                if not params.do_not_save:
                    saver.save(sess, ckpt_dir + 'model.ckpt', epoch_n)

                # ---------------------------------------------
                # Compute error on testing set (10k examples)
                test_costs = evaluate_metric_list(mnist.test, sess, test_losses)
                train_costs = sess.run(
                    train_losses,
                    feed_dict={inputs_placeholder: images,
                               outputs: labels,
                               train_flag: False})

                # Create log of:
                # time, epoch number, test accuracy, test cross entropy,
                # train accuracy, train loss, train cross entropy,
                # train reconstruction loss

                log_i = [now, epoch_n] + sess.run(
                    [accuracy],
                    feed_dict={inputs_placeholder: mnist.test.images,
                               outputs: mnist.test.labels,
                               train_flag: False}
                ) + test_costs + sess.run(
                    [accuracy],
                    feed_dict={inputs_placeholder:
                                   mnist.train.labeled_ds.images,
                               outputs: mnist.train.labeled_ds.labels,
                               train_flag: False}
                ) + train_costs

                with open(log_file, 'a') as train_log:
                    print(*log_i, sep=',', flush=True, file=train_log)

    with open(desc_file, 'a') as f:
        print("Final Accuracy: ", sess.run(accuracy, feed_dict={
            inputs_placeholder: mnist.test.images, outputs: mnist.test.labels,
            train_flag: False}),
              "%", file=f, flush=True)


    sess.close()


if __name__ == '__main__':
    main()








