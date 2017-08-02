import tensorflow as tf
import numpy as np
import IPython
from conv_ladder import *

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
    
    batch_size = 100
    x = tf.placeholder(shape=[batch_size, 32, 32, 3], dtype=tf.float32)

    enc_out = encoder(x)
    dec_out = decoder(enc_out)



if __name__ == '__main__':
    # test_cnn_ladder()
    make_layer_spec()