
from source import *
from math import floor
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)


def make_feed(images, labels):
    return {x: images, y: labels}

# ===========================
# PARAMETERS
# ===========================
BATCH_SIZE = 100
INPUT_SIZE = 784
TRAIN_FLAG = True
OUTPUT_SIZE = 10
EX_ID = 'mlp_gamma_all'
N = mnist.train.num_examples
# MAX_ITER = 100
MAX_EPOCHS = 100


iter_per_epoch = int(N / BATCH_SIZE)
MAX_ITER = MAX_EPOCHS * iter_per_epoch


# ===========================
# ENCODER
# ===========================
# Start with a tuple specifying layer sizes
enc_layers = [INPUT_SIZE, 1000, 500, 250, 250, 250, OUTPUT_SIZE]

# Input placeholder
x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_SIZE))
# One-hot targets
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_SIZE))


# CLEAN ENCODER
clean = Encoder(x, y, enc_layers, noise_sd=None, reuse=False, training=TRAIN_FLAG)

# ===========================
# CORRUPTED ENCODER
noisy = Encoder(x, y, enc_layers, noise_sd=0.3, reuse=True, training=TRAIN_FLAG)

# Do we share batch norm weights?

# ===========================
# GAMMA DECODER
# ===========================
# Each of these are BATCH_SIZE x OUTPUT_SIZE (or more generally enc_layers[l])
L = clean.n_layers
l = L - 1

# Batch normalized signal from layer above in decoder (v)
if l < noisy.last:
    v_L = noisy.h[l + 1]
else:
    v_L = tf.expand_dims(tf.cast(noisy.predict, tf.float32), axis=-1)  # label, with dim matching
# Use decoder weights to upsample the signal from above
size_out = enc_layers[l]
u_L = fclayer(v_L, size_out, scope='dec'+str(l))

# Unbatch-normalized activations from parallel layer in noisy encoder
z_L = noisy.z[l]

# Unbatch-normalized target activations from parallel layer in clean encoder
target_z = clean.z[l]

uz_L = tf.multiply(u_L, z_L)

inputs = tf.stack([u_L, z_L, uz_L], axis=-1)
combinator = Combinator(inputs, layer_sizes=(2,2,1), stddev=0.025)

recons = combinator.outputs

rc_cost = tf.reduce_sum(tf.square(noisy.bn_layers[l].normalize_from_saved_stats(recons) - target_z), axis=-1)



# test_dict = make_feed(*mnist.test.next_batch(BATCH_SIZE))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     outs = sess.run([target_z, u_L, z_L, uz_L, inputs, recons, rc_cost, noisy.loss], test_dict)
#     print([x.shape for x in outs])
#     # outs = sess.run(, test_dict)
#     # print(outs.shape)



# ===========================
# TRAINING
# ===========================
# loss = clean.loss
# err_op = 1 - tf.reduce_mean(tf.cast(tf.equal(clean.predict, tf.argmax(y, 1)), tf.float32))
loss = noisy.loss + rc_cost
err_op = 1 - tf.reduce_mean(tf.cast(tf.equal(noisy.predict, tf.argmax(y, 1)), tf.float32))

# Set up decay learning rate
global_step = tf.Variable(0, trainable=False)
lr_init = 0.0002
decay_start = MAX_ITER
decay_duration = 50
decay_step = lr_init/decay_duration
decay_end = decay_start + decay_duration
boundaries = [b for b in range(MAX_ITER, decay_end, 1)]
values = [lr_init - (t * decay_step) for t in range(decay_duration+1)]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

# Passing global_step to minimize() will increment it at each step.
opt_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

saver = tf.train.Saver()
save_to = 'models/' + EX_ID
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter('logs/' + EX_ID + '/train/', sess.graph)
test_writer = tf.summary.FileWriter('logs/' + EX_ID + '/test/', sess.graph)
tf.summary.scalar('loss', tf.reduce_mean(loss))
tf.summary.scalar('error', err_op)
merged = tf.summary.merge_all()

test_dict = make_feed(*mnist.test.next_batch(BATCH_SIZE))
print('Epoch', 'Step', 'TrainErr(%)', 'TestErr(%)', sep='\t')
# Training
for step in range(decay_end):
    train_dict = make_feed(*mnist.train.next_batch(BATCH_SIZE))
    sess.run(opt_op, feed_dict=train_dict)
    # Logging during training

    epoch = floor(step * BATCH_SIZE / mnist.train.num_examples)

    if step % 100 == 0:
        train_summary, train_acc = \
            sess.run([merged, err_op], train_dict)
        test_summary, test_acc = \
            sess.run([merged, err_op], test_dict)
        train_writer.add_summary(train_summary, global_step=step)
        test_writer.add_summary(test_summary, global_step=step)

        print(epoch, step, train_acc*100, test_acc*100, sep='\t')

# Save final model
saved_to = saver.save(sess, save_to)
print('Model saved to: ', saved_to)
