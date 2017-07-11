
import argparse

from tensorflow.examples.tutorials.mnist import input_data

from src import feed
from src.ladder import *

mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

def to_steps(epoch):
    return epoch * ITER_PER_EPOCH

def to_epochs(step):
    return step / ITER_PER_EPOCH

def decay_learning_rate(initial_learning_rate, decay_start_epoch, end_epoch, global_step):
    end_step = to_steps(end_epoch)
    decay_start_step = to_steps(decay_start_epoch)

    decay_epochs = end_epoch - decay_start_epoch
    boundaries = [x for x in range(decay_start_step, end_step, ITER_PER_EPOCH)]
    decay_per_epoch = initial_learning_rate / decay_epochs
    values = [initial_learning_rate - x * decay_per_epoch for x in range(decay_epochs + 1)]
    assert len(values) == len(boundaries) + 1

    return tf.train.piecewise_constant(global_step, boundaries, values)


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

params = parser.parse_args()

# TRAIN_BATCH_SIZE = params.labeled_batch_size + params.unlabeled_batch_size
TRAIN_FLAG = params.training
EX_ID = params.id
ITER_PER_EPOCH = int(params.num_labeled / params.labeled_batch_size)

# Specify base structure
INPUT_SIZE = 784
OUTPUT_SIZE = 10
LAYER_SIZES = [INPUT_SIZE, 1000, 500, 250, 250, 250, OUTPUT_SIZE]
SC_WEIGHT = 1000
RC_WEIGHTS = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:1.0}


# Set up decaying learning rate
global_step = tf.get_variable('global_step', initializer=0, trainable=False)
learning_rate = decay_learning_rate(
    params.initial_learning_rate,
    params.decay_start_epoch,
    params.end_epoch,
    global_step)


param_dict = {'layer_sizes': LAYER_SIZES,
              'train_flag': TRAIN_FLAG,
              'labeled_batch_size': params.labeled_batch_size,
              'gamma_flag': params.gamma,
              'unlabeled_batch_size': params.unlabeled_batch_size,
              'sc_weight': SC_WEIGHT,
              'rc_weights': RC_WEIGHTS}

ladder = Ladder(param_dict)

# ===========================
# TRAINING
# ===========================
# Passing global_step to minimize() will increment it at each step.
opt_op = tf.train.AdamOptimizer(learning_rate).minimize(ladder.loss, global_step=global_step)

# Set up saver
saver = tf.train.Saver()
save_to = 'models/' + EX_ID

# Start session and initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create summaries
train_writer = tf.summary.FileWriter('logs/' + EX_ID + '/train/', sess.graph)
tf.summary.scalar('loss', ladder.mean_loss)
tf.summary.scalar('error', ladder.aer)
merged = tf.summary.merge_all()


sf = feed.MNIST(mnist.train.images, mnist.train.labels, params.num_labeled)

print('Labeled Epoch', 'Unlabeled Epoch', 'Step', 'Loss', 'TrainErr(%)', sep='\t', flush=True)
# Training (using a separate step to count)
end_step = to_steps(params.end_epoch)
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
