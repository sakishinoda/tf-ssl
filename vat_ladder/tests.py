import tensorflow as tf
import numpy as np
import os
import IPython

from tensorflow.examples.tutorials.mnist import input_data
import argparse

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

def test_simple_dae(training=True):
    """Test of encoder/decoder in a simple denoising set up"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    nll_wt = tf.placeholder(shape=(), dtype=tf.float32)

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

    if training:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(100):
                images, labels = mnist.train.next_batch(batch_size)
                _, step_aer, step_loss, step_nll, step_rc = sess.run([train_op, aer, loss, nll, rc], feed_dict={x_vec: images, y: labels, nll_wt: 1000.0*(0.99**step)})
                print(step, step_aer, step_loss, step_nll, step_rc, sep='\t', flush=True)

            saver.save(sess, 'test.ckpt')

    else:
        with tf.Session() as sess:
            saver.restore(sess, 'test.ckpt')
            num_steps = mnist.test.num_examples//batch_size
            mean_aer = 0
            for step in range(num_steps):
                images, labels = mnist.test.next_batch(batch_size)
                step_aer = sess.run(aer, feed_dict={x_vec: images, y: labels, nll_wt: 1.0})
                print(step, step_aer)
                mean_aer += step_aer

            print("Mean AER:", mean_aer/num_steps)

def test_nargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static_bn', default=False, nargs='?', const=0.99,
                        type=float)
    params = parser.parse_args()
    if params.static_bn is False:
        bn_decay = tf.Variable(1e-10, trainable=False)
    else:
        bn_decay = params.static_bn

    print(bn_decay)

from src.val import build_graph
from src.utils import get_cli_params, process_cli_params
from src import mnist
from src.val import softmax_cross_entropy_with_logits
def test_hessian_ops():
    p = process_cli_params(get_cli_params())
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
    p.model = 'ladder'

    # Build graph
    inputs = tf.placeholder(tf.float32, shape=(784))
    outputs = tf.placeholder(tf.float32)
    logits = tf.matmul(
        tf.expand_dims(inputs, axis=0), tf.get_variable('w', shape=(784, 10))
    )


    loss = softmax_cross_entropy_with_logits(outputs, logits)
    loss_grad = tf.gradients(loss, inputs, name='help')
    # hess = tf.hessians(loss, inputs)

    IPython.embed()
    # ul_hess = tf.hessians(g['ladder'].u_cost, g['images'])

    # s = tf.stop_gradient(tf.svd(hess, compute_uv=False))
    # ul_s = tf.stop_gradient(tf.svd(ul_hess, compute_uv=False))

    # print(s.get_shape(), ul_s.get_shape())





if __name__ == '__main__':

    from vat_ladder import test_data_splitting
    test_data_splitting()

    # from src.svhn import read_data_sets
    # svhn = read_data_sets('../../data/svhn/')
    # IPython.embed()

    # test_hessian_ops()
    # test_nargs()
    # test_simple_dae(training=False)
    # test_cnn_ladder()
    # print(*make_layer_spec().items(), sep='\n')
    # make_layer_spec()