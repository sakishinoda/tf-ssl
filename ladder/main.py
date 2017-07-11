
import argparse

from tensorflow.examples.tutorials.mnist import input_data

from src import feed, utils
from src.ladder import *

mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

# ===========================
# PARAMETERS
# ===========================

parser = argparse.ArgumentParser()
parser.add_argument('--id', default='ladder')
parser.add_argument('--training', action='store_true')
parser.add_argument('--decay_start_epoch', default=100, type=int)
parser.add_argument('--end_epoch', default=150, type=int)
parser.add_argument('--print_interval', default=100, type=int)
parser.add_argument('--save_interval', default=None, type=int)
parser.add_argument('--num_labeled', default=100, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--labeled_batch_size', default=100, type=int)
parser.add_argument('--unlabeled_batch_size', default=250, type=int)
parser.add_argument('--initial_learning_rate', default=0.0002, type=float)
parser.add_argument('--gamma', action='store_true')
parser.add_argument('--layer_sizes', default='784-1000-500-250-250-250-10')
parser.add_argument('--sc_weight', default=1000, type=float)
parser.add_argument('--rc_weights', default='0-0-0-0-0-0-1')

params = parser.parse_args()

# Specify base structure
iter_per_epoch = int(params.num_labeled / params.labeled_batch_size)
# input_size = 784
# output_size = 10
# layer_sizes = [input_size, 1000, 500, 250, 250, 250, output_size]
layer_sizes = utils.parse_argstring(params.layer_sizes)
sc_weight = params.sc_weight
rc_weights = utils.parse_argstring(params.rc_weights, dtype=float)
rc_weights = zip(range(len(rc_weights)), rc_weights)

# Set up decaying learning rate
global_step = tf.get_variable('global_step', initializer=0, trainable=False)
learning_rate = utils.decay_learning_rate(
    initial_learning_rate=params.initial_learning_rate,
    decay_start_epoch=params.decay_start_epoch,
    end_epoch=params.end_epoch,
    iter_per_epoch=iter_per_epoch,
    global_step=global_step)

param_dict = {'layer_sizes': layer_sizes,
              'train_flag': params.training,
              'labeled_batch_size': params.labeled_batch_size,
              'gamma_flag': params.gamma,
              'unlabeled_batch_size': params.unlabeled_batch_size,
              'sc_weight': sc_weight,
              'rc_weights': rc_weights}

ladder = Ladder(param_dict)

# ===========================
# TRAINING
# ===========================
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


sf = feed.MNIST(mnist.train.images, mnist.train.labels, params.num_labeled)

print('Labeled Epoch', 'Unlabeled Epoch', 'Step', 'Loss', 'TrainErr(%)', sep='\t', flush=True)
# Training (using a separate step to count)
end_step = params.end_epoch * iter_per_epoch
for step in range(end_step):
    x, y = sf.next_batch(params.labeled_batch_size, params.unlabeled_batch_size)
    train_dict = {ladder.x: x, ladder.y: y}
    sess.run(opt_op, feed_dict=train_dict)

    # Logging during training
    if (step+1) % params.print_interval == 0:
        labeled_epoch = sf.labeled.epochs_completed
        unlabeled_epoch = sf.unlabeled.epochs_completed
        train_summary, train_err, train_loss = \
            sess.run([merged, ladder.aer, ladder.mean_loss], train_dict)
        train_writer.add_summary(train_summary, global_step=step)

        print(labeled_epoch, unlabeled_epoch, step, train_loss, train_err * 100, sep='\t', flush=True)

    if params.save_interval is not None and step % params.save_interval == 0:
        saver.save(sess, save_to, global_step=step)

    global_step += 1

# Save final model
saved_to = saver.save(sess, save_to)
print('Model saved to: ', saved_to)
