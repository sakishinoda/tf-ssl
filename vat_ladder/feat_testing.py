import numpy as np
import tensorflow as tf
import IPython

def amlp_combinator(z_c, u, size):
    w_init = [[1., 1., 1., 1.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]]
    w_init = np.asarray(w_init)
    uz = tf.multiply(z_c, u)
    x = tf.stack([z_c, u, uz], axis=-1)
    x = tf.reshape(x, shape=[-1, 3])

    res = tf.matmul(x, w_init)
    return res


def main():
    z = tf.placeholder_with_default(np.random.randn([16, 20]))
    u = tf.placeholder_with_default(np.random.randn([16, 20]))

    res = amlp_combinator(z, u, None)

    IPython.embed()


if __name__ == '__main__':
    main()




