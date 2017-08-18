
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


def build_graph(params):

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
    adv = Adversary(ladder.bn, params)
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
    vat_cost = adv.virtual_adversarial_loss(
        x=unlabeled(inputs),
        logit=unlabeled(ladder.corr.logits),
        is_training=train_flag,
        start_layer=0) * params.vat_weight

    # -----------------------------
    # Loss, accuracy and training steps
    loss = ladder.cost + ladder.u_cost + vat_cost
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

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5,
                            max_to_keep=5)

    # Graph
    g = dict()
    g['images'] = inputs_placeholder
    g['labels'] = outputs
    g['train_flag'] = train_flag
    g['ladder'] = ladder
    g['saver'] = saver
    g['train_step'] = train_step
    g['lr'] = learning_rate

    # Metrics
    m = dict()
    m['loss'] = loss
    m['cost'] = ladder.cost
    m['uc'] = ladder.u_cost
    m['vc'] = vat_cost
    m['acc'] = accuracy

    trainable_params = count_trainable_params()

    return g, m, trainable_params

def evaluate_metric(dataset, sess, op, graph, params):
    metric = 0
    num_eval_iters = dataset.num_examples // params.eval_batch_size
    for _ in range(num_eval_iters):
        images, labels = dataset.next_batch(params.eval_batch_size)
        init_feed = {graph.inputs_placeholder: images,
                     graph.outputs: labels,
                     graph.train_flag: False}
        metric += sess.run(op, init_feed)
    metric /= num_eval_iters
    return metric

def evaluate_metric_list(dataset, sess, ops, graph, params):
    metrics = [0.0 for _ in ops]
    num_eval_iters = dataset.num_examples // params.eval_batch_size
    for _ in range(num_eval_iters):
        images, labels = dataset.next_batch(params.eval_batch_size)
        init_feed = {graph.inputs_placeholder: images,
                     graph.outputs: labels,
                     graph.train_flag: False}
        op_eval = sess.run(ops, init_feed)

        for i, op in enumerate(op_eval):
            metrics[i] += op

    metrics = [metric/num_eval_iters for metric in metrics]
    return metrics


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
                                      validation_size=p.validation_size,
                                      one_hot=True,
                                      disjoint=False)
    num_examples = mnist.train.num_examples
    p.num_examples = num_examples
    if p.validation:
        mnist.test = mnist.validation
    p.iter_per_epoch = (num_examples // p.batch_size)
    p.num_iter = p.iter_per_epoch * p.end_epoch

    # Build graph
    g, m, trainable_parameters = build_graph(p)

    # Collect losses
    train_losses = [m['loss'], m['cost'], m['uc'], m['vc']]
    test_losses = [m['cost']]

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
        # if checkpoint exists, restore the parameters and set epoch_n and i_iter
        g['saver'].restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
        i_iter = (epoch_n + 1) * (p.num_examples // p.batch_size)
        print("Restored Epoch ", epoch_n)
    else:
        # no checkpoint exists. create checkpoints directory if it does not exist.
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
        print("Initial Train Losses: ", *eval_metrics(
            mnist.train, sess, train_losses), file=f,
              flush=True)

        # -----------------------------
        # Evaluate initial testing accuracy and cross-entropy loss
        print("Initial Test Accuracy: ",
              sess.run(m['acc'], feed_dict={
                  g['in']: mnist.test.images,
                  g['labels']: mnist.test.labels,
                  g['train_flag']: False}),
              "%", file=f, flush=True)
        print("Initial Test Losses: ",
              *eval_metrics(
                  mnist.test, sess, test_losses), file=f,
              flush=True)

    start = time.time()
    for i in tqdm(range(i_iter, p.num_iter)):

        images, labels = mnist.train.next_batch(p.batch_size)

        _ = sess.run(
            [g['train_step']],
            feed_dict={g['images']: images,
                       g['labels']: labels,
                       g['train_flag']: True})

        # ---------------------------------------------
        # Epoch completed?
        if (i > 1) and ((i + 1) % p.iter_per_epoch == 0):
            epoch_n = i // (p.num_examples // p.batch_size)

            # ---------------------------------------------
            # Update batch norm decay constant
            if p.bn_decay == 'dynamic':
                g['ladder'].bn_decay.assign(1.0 - (1.0 / (epoch_n + 1)))

            # ---------------------------------------------
            # Update learning rate every epoch
            if (epoch_n + 1) >= p.decay_start_epoch:
                # epoch_n + 1 because learning rate is set for next epoch
                ratio = 1.0 * (p.end_epoch - (epoch_n + 1))
                ratio = max(0., ratio / (p.end_epoch - p.decay_start_epoch))
                sess.run(g['lr'].assign(p.initial_learning_rate *
                                                ratio))

            # ---------------------------------------------
            # Evaluate every test_frequency_in_epochs
            if ((i + 1) % (p.test_frequency_in_epochs *
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
                # train reconstruction loss

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








