import tensorflow as tf
import numpy as np
import IPython
from conv_ladder import *
from tensorflow.examples.tutorials.mnist import input_data

def test_copying_by_strided_deconv():
    x = tf.placeholder(dtype=tf.float32, shape=[10,4,4,1])
    w = tf.ones(dtype=tf.float32, shape=[2,2,1,1])
    output_shape = x.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[2] *= 2
    h = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1,2,2,1],
                               padding='SAME', data_format='NHWC')
    # g = tf.nn.atrous_conv2d(x, w, rate=2, padding='SAME')
    g = tf.squeeze(h)
    # g = tf.nn.atrous_conv2d(x, w, rate=0.5, padding='SAME')

    x_fill = np.random.randint(1, 10, size=[10,4,4,1])

    with tf.Session() as sess:
        [h_eval, g_eval] = sess.run([h, g], feed_dict={x: x_fill})
        print(x_fill, h_eval, g, sep='\n')
        IPython.embed()


def test_cnn_ladder():
    batch_size = 100
    x = tf.placeholder(shape=[batch_size, 32, 32, 3], dtype=tf.float32)

    enc_out = encoder(x)
    dec_out = decoder(enc_out)

    print(enc_out.get_shape(), dec_out.get_shape())

def test_simple_dae():
    """Test of encoder/decoder in a simple denoising set up"""
    mnist = input_data.read_data_sets("MNIST_data",
                                      one_hot=True)
    layers = make_layer_spec(
        types=(
        'c', 'c', 'c', 'max', 'c', 'c', 'c', 'max', 'c', 'c', 'c', 'avg', 'fc'),
        fan=(1, 96, 96, 96, 96, 192, 192, 192, 192, 192, 192, 192, 192, 10),
        ksizes=(3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, None, None),
        strides=(1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, None, None),
        init_size=28
    )

    batch_size = 100
    x_vec = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32)
    x = tf.reshape(x_vec, [batch_size, 28, 28, 1])
    y = tf.placeholder(shape=[batch_size, 10], dtype=tf.float32)
    nll_wt = tf.placeholder_with_default(input=10.0)

    enc_out = encoder(x, layers=layers)
    nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=enc_out, labels=y), axis=[0])
    predict = tf.argmax(enc_out, axis=-1)
    aer = 1 - tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(y, axis=1)), tf.float32))
    dec_out = decoder(enc_out, layers=layers)
    rc = tf.reduce_mean(tf.reduce_sum(tf.square(dec_out - x), axis=[1,2,3]), axis=[0])

    loss = nll_wt * nll + rc
    print(nll.get_shape(), rc.get_shape())

    train_op = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(500):
            images, labels = mnist.train.next_batch(batch_size)
            _, step_aer, step_loss, step_nll, step_rc = sess.run([train_op, aer, loss, nll, rc], feed_dict={x_vec: images, y: labels, nll_wt: 100.0*(0.9**step)})
            print(step, step_aer, step_loss, step_nll, step_rc, sep='\t', flush=True)

        saver.save(sess, 'test.ckpt')



if __name__ == '__main__':
    test_simple_dae()
    # test_cnn_ladder()
    # print(*make_layer_spec().items(), sep='\n')
    # make_layer_spec()