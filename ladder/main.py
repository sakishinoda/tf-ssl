# import IPython
from tensorflow.examples.tutorials.mnist import input_data
from time import time
import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))
from src import feed, utils
from src.ladder import *

# ===========================
# PARAMETERS
# ===========================
params = utils.get_cli_params()
write_to = open(params.write_to, 'w') if params.write_to is not None else None

param_dict = vars(params)
print('===== Parameter settings =====', flush=True, file=write_to)
sorted_keys = sorted([k for k in param_dict.keys()])
for k in sorted_keys:
    print(k, ':', param_dict[k], file=write_to, flush=True)

params = utils.process_cli_params(params)

# Specify GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(params.which_gpu)

mnist = input_data.read_data_sets(sys.path[0]+'/../data/mnist/', one_hot=True)

ladder = Ladder(params)
# IPython.embed()


if params.train_flag:
    # ===========================
    # TRAINING
    # ===========================
    sf = feed.Balanced(mnist.train.images, mnist.train.labels, params.num_labeled)
    print('seeds : ', sf.seeds, file=write_to, flush=True)
    iter_per_epoch = int(sf.unlabeled.num_examples / params.unlabeled_batch_size)
    print('iter_per_epoch', ':', iter_per_epoch, file=write_to, flush=True)
    utils.print_trainables(write_to=write_to)
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
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    save_to = 'models/' + params.id

    # Start session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create summaries
    train_writer = tf.summary.FileWriter('logs/' + params.id + '/train/', sess.graph)
    test_writer = tf.summary.FileWriter('logs/' + params.id + '/test/', sess.graph)
    train_loss_summary = tf.summary.scalar('training loss', ladder.loss)
    test_loss_summary = tf.summary.scalar('testing loss', ladder.testing_loss)
    err_summary = tf.summary.scalar('error', ladder.aer)
    train_merged = tf.summary.merge([train_loss_summary, err_summary])
    test_merged = tf.summary.merge([test_loss_summary, err_summary])

    x, y = mnist.validation.next_batch(params.labeled_batch_size)
    test_dict = {ladder.x: x, ladder.y: y}
    start_time = time()

    print('LEpoch', 'UEpoch', 'Time/m', 'Step', 'Loss', 'TestLoss', 'AER(%)', 'TestAER(%)', sep='\t', flush=True, file=write_to)
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

            # Training metrics
            train_summary, train_err, train_loss = \
                sess.run([train_merged, ladder.aer, ladder.loss],
                         feed_dict=train_dict)
            train_writer.add_summary(train_summary, global_step=step)

            # Testing metrics
            test_summary, test_err, test_loss = \
                sess.run([test_merged, ladder.aer, ladder.testing_loss],
                         feed_dict=test_dict)
            test_writer.add_summary(test_summary, global_step=step)

            print(
                labeled_epoch,
                unlabeled_epoch,
                int(time_elapsed/60),
                step,
                train_loss,
                test_loss,
                train_err * 100,
                test_err * 100,
                sep='\t', flush=True, file=write_to)


        if save_interval is not None and step > 0 and step % save_interval == 0:
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
    # sess.run(tf.global_variables_initializer())

    saver.restore(sess, save_to)

    err = []
    loss = []

    for step in range(iter_per_epoch):
        x, y = test_set.next_batch(params.test_batch_size)
        feed_dict = {ladder.x: x, ladder.y: y}

        this_err, this_loss = \
            sess.run([ladder.aer, ladder.testing_loss], feed_dict)
        err.append(this_err)
        loss.append(this_loss)

        if params.verbose:
            print(step, loss[step], err[step] * 100, sep='\t', flush=True, file=write_to)

    mean_loss = sum(loss) / (len(loss)+1)
    mean_aer = sum(err) / (len(err)+1)

    print('Training step', 'Mean loss', 'Mean AER(%)', sep='\t', flush=True, file=write_to)
    print(params.train_step, mean_loss, mean_aer * 100, sep='\t', flush=True, file=write_to)

sess.close()
if write_to is not None:
    write_to.close()