
import argparse

from tensorflow.examples.tutorials.mnist import input_data

from src import feed
from src.ladder import *

mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

def make_feed(images, labels):
    return {x: images, y: labels}

def to_steps(epoch):
    return epoch * ITER_PER_EPOCH

def to_epochs(step):
    return step / ITER_PER_EPOCH

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
parser.add_argument('--gamma', action='store_true')

params = parser.parse_args()

NUM_LABELED = params.num_labeled
LABELED_BATCH_SIZE = params.labeled_batch_size
UNLABELED_BATCH_SIZE = params.unlabeled_batch_size
TRAIN_BATCH_SIZE = LABELED_BATCH_SIZE + UNLABELED_BATCH_SIZE
TEST_BATCH_SIZE = LABELED_BATCH_SIZE


INPUT_SIZE = 784
TRAIN_FLAG = params.training
OUTPUT_SIZE = 10
EX_ID = params.id
PRINT_INTERVAL = params.print_interval
SAVE_INTERVAL = params.save_interval
USE_GAMMA_DECODER = params.gamma
NUM_EXAMPLES = mnist.train.num_examples


# Specify base structure
LAYER_SIZES = [INPUT_SIZE, 1000, 500, 250, 250, 250, OUTPUT_SIZE]
SC_WEIGHT = 1000
RC_WEIGHTS = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:1.0}

DECAY_START_EPOCH = params.decay_start_epoch
END_EPOCH = params.end_epoch
DECAY_EPOCHS = END_EPOCH - DECAY_START_EPOCH
ITER_PER_EPOCH = int(NUM_EXAMPLES / TRAIN_BATCH_SIZE)
DECAY_START_STEP = to_steps(DECAY_START_EPOCH)
END_STEP = to_steps(END_EPOCH)


# Set up decaying learning rate
INITIAL_LEARNING_RATE = 0.0002
boundaries = [x for x in range(DECAY_START_STEP, END_STEP, ITER_PER_EPOCH)]
DECAY_PER_EPOCH = INITIAL_LEARNING_RATE / DECAY_EPOCHS
values = [INITIAL_LEARNING_RATE - x * DECAY_PER_EPOCH for x in range(DECAY_EPOCHS + 1)]
assert len(values) == len(boundaries) + 1

global_step = tf.get_variable('global_step', initializer=0, trainable=False)
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

# ===========================
# ENCODER
# ===========================

# Input placeholder
x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
# One-hot targets
y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))

# CLEAN ENCODER
clean = Encoder(x, y, LAYER_SIZES, noise_sd=None, reuse=False, training=TRAIN_FLAG)

# ===========================
# CORRUPTED ENCODER
noisy = Encoder(x, y, LAYER_SIZES, noise_sd=0.3, reuse=True, training=TRAIN_FLAG)

# Compute supervised loss on labeled only
supervised_loss = tf.concat((noisy.supervised_loss(LABELED_BATCH_SIZE),
                             tf.zeros((UNLABELED_BATCH_SIZE,))), axis=0) * SC_WEIGHT

# Compute training error rate on labeled examples only (since e.g. CIFAR-100 with Tiny Images, no labels are actually available)
avg_err_rate = 1 - tf.reduce_mean(tf.cast(tf.equal(noisy.predict[:LABELED_BATCH_SIZE], tf.argmax(y, 1)), tf.float32))
# ===========================
# DECODER
# ===========================
if USE_GAMMA_DECODER:
    decoder = GammaDecoder(noisy, clean)
else:
    decoder = Decoder(noisy, clean)

decoder.build(tf.expand_dims(tf.cast(noisy.predict, tf.float32), axis=-1))
unsupervised_loss = decoder.unsupervised_loss(RC_WEIGHTS)
# ===========================
# TRAINING
# ===========================
# Total loss
total_loss = supervised_loss + unsupervised_loss

# Passing global_step to minimize() will increment it at each step.
opt_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(total_loss, global_step=global_step)

# Set up saver
saver = tf.train.Saver()
save_to = 'models/' + EX_ID

# Start session and initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create summaries
train_writer = tf.summary.FileWriter('logs/' + EX_ID + '/train/', sess.graph)
test_writer = tf.summary.FileWriter('logs/' + EX_ID + '/test/', sess.graph)
tf.summary.scalar('loss', tf.reduce_mean(total_loss))
tf.summary.scalar('error', avg_err_rate)
merged = tf.summary.merge_all()

test_dict = make_feed(*mnist.validation.next_batch(TEST_BATCH_SIZE))
sf = feed.MNIST(mnist.train.images, mnist.train.labels, params.num_labeled)
print('Epoch', 'Step', 'TrainErr(%)', 'TestErr(%)', sep='\t', flush=True)
# Training (using a separate step to count)
for step in range(END_STEP):
    train_dict = make_feed(*sf.next_batch(LABELED_BATCH_SIZE, UNLABELED_BATCH_SIZE))
    sess.run(opt_op, feed_dict=train_dict)
    # Logging during training

    epoch = mnist.train.epochs_completed

    if (step+1) % params.print_interval == 0:
        train_summary, train_err = \
            sess.run([merged, avg_err_rate], train_dict)
        test_err = 0
        # test_summary, test_err = \
        #     sess.run([merged, avg_err_rate], test_dict)
        train_writer.add_summary(train_summary, global_step=step)
        # test_writer.add_summary(test_summary, global_step=step)

        print(epoch, step, train_err * 100, test_err * 100, sep='\t', flush=True)

    if params.save_interval is not None and step % params.save_interval == 0:
        saver.save(sess, save_to, global_step=step)

    global_step += 1

# Save final model
saved_to = saver.save(sess, save_to)
print('Model saved to: ', saved_to)

# ===========================
# TEST
# ===========================
test_steps = int(mnist.test.num_examples / TEST_BATCH_SIZE)
test_err = 0
for step in range(test_steps):
    test_dict = make_feed(*mnist.test.next_batch(TEST_BATCH_SIZE))
    test_err += sess.run(avg_err_rate, test_dict)
print('Final test error (%):', 100 * test_err / test_steps, flush=True)