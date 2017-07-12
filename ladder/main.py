from tensorflow.examples.tutorials.mnist import input_data
from time import time
import sys, os
# sys.path.append('/Users/saki/tf-ssl/')
sys.path.append(os.path.join(sys.path[0],'..'))
from src import feed, utils
from src.ladder import *

# Specify GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ===========================
# PARAMETERS
# ===========================
params = utils.get_cli_params()

mnist = input_data.read_data_sets(sys.path[0]+'/../data/mnist/', one_hot=True)

ladder = Ladder(params)


if params.train_flag:

    # ===========================
    # TRAINING
    # ===========================
    sf = feed.MNIST(mnist.train.images, mnist.train.labels, params.num_labeled)
    iter_per_epoch = int(sf.unlabeled.num_examples / params.unlabeled_batch_size)
    print('iter_per_epoch', ':', iter_per_epoch)
    utils.print_trainables()
    save_interval = None if params.save_epochs is None else int(params.save_epochs * iter_per_epoch)


    global_step = tf.get_variable('global_step', initializer=0, trainable=False)
    learning_rate = utils.decay_learning_rate(
            initial_learning_rate=params.initial_learning_rate,
            decay_start_epoch=params.decay_start_epoch,
            end_epoch=params.end_epoch,
            iter_per_epoch=iter_per_epoch,
            global_step=global_step)

    # Passing global_step to minimize() will increment it at each step.
    opt_op = tf.train.AdamOptimizer(learning_rate).minimize(ladder.loss, global_step=global_step)

    # Set up saver
    saver = tf.train.Saver()
    save_to = 'models/' + params.id

    # Start session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create summaries
    train_writer = tf.summary.FileWriter('logs/' + params.id + '/train/', sess.graph)
    tf.summary.scalar('loss', ladder.mean_loss)
    tf.summary.scalar('error', ladder.aer)
    merged = tf.summary.merge_all()

    start_time = time()

    print('LEpoch', 'UEpoch', 'Time/m', 'Step', 'Loss', 'AER(%)', sep='\t', flush=True)
    # Training (using a separate step to count)
    end_step = params.end_epoch * iter_per_epoch
    for step in range(end_step):
        x, y = sf.next_batch(params.labeled_batch_size, params.unlabeled_batch_size)
        train_dict = {ladder.x: x, ladder.y: y}
        sess.run(opt_op, feed_dict=train_dict)

        # Logging during training
        if (step+1) % params.print_interval == 0:
            time_elapsed = time() - start_time
            labeled_epoch = sf.labeled.epochs_completed
            unlabeled_epoch = sf.unlabeled.epochs_completed
            train_summary, train_err, train_loss = \
                sess.run([merged, ladder.aer, ladder.mean_loss], train_dict)
            train_writer.add_summary(train_summary, global_step=step)

            print(
                labeled_epoch,
                unlabeled_epoch,
                int(time_elapsed/60),
                step,
                train_loss,
                train_err * 100,
                sep='\t', flush=True)

        if save_interval is not None and step % save_interval == 0:
            saver.save(sess, save_to, global_step=step)

        global_step += 1

    # Save final model
    saved_to = saver.save(sess, save_to)
    print('Model saved to: ', saved_to)

else:

    # ===========================
    # TESTING
    # ===========================

    test_set = mnist.test
    iter_per_epoch = int(test_set.num_examples / params.num_labeled)

    # Set up saver
    saver = tf.train.Saver()
    save_to = 'models/' + params.id
    if params.train_step is not None:
        save_to += '-' + str(params.train_step)

    # Start session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, save_to)


    err = [0 for x in range(iter_per_epoch)]
    loss = [0 for x in range(iter_per_epoch)]


    for step in range(iter_per_epoch):
        x, y = test_set.next_batch(params.test_batch_size)
        feed_dict = {ladder.x: x, ladder.y: y}

        err[step], loss[step] = \
            sess.run([ladder.aer, ladder.mean_loss], feed_dict)

        if params.verbose:
            print(step, loss[step], err[step] * 100, sep='\t', flush=True)

    mean_loss = sum(loss) / (iter_per_epoch)
    mean_aer = sum(err) / (iter_per_epoch)

    print('Training step', 'Mean loss', 'Mean AER(%)', sep='\t', flush=True)
    print(params.train_step, mean_loss, mean_aer * 100, sep='\t', flush=True)

sess.close()