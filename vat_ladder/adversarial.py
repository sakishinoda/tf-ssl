from cleverhans.model import Model as CleverHansModel
from cleverhans.utils_tf import model_eval
from src.val import build_graph_from_inputs, build_graph
from src.utils import get_cli_params, process_cli_params
from src.mnist import read_data_sets
import tensorflow as tf


class MyModel(CleverHansModel):
    def __init__(self, params, scope='model'):
        super(MyModel, self).__init__()
        self.params = params
        self.scope = scope
        with tf.variable_scope(scope, reuse=None):
            self.g, self.m, _ = build_graph(params, is_training=False)


    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        # return self.get_layer(x, 'logits')
        with tf.variable_scope(self.scope, reuse=True):
            g, m, _ = build_graph_from_inputs(x, self.params, is_training=False)
            return g['model'].clean.logits

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        with tf.variable_scope(self.scope, reuse=True):
            g, m, _ = build_graph_from_inputs()
            return g['model'].clean.labeled.h[-1]


def main():
    params = process_cli_params(get_cli_params())
    model = MyModel(params)
    dataset = read_data_sets("MNIST_data",
                             n_labeled=p.num_labeled,
                             validation_size=p.validation,
                             one_hot=True,
                             disjoint=False)



    with tf.Session() as sess:

        id_seed_dir = p.id + "/" + "seed-{}".format(p.seed) + "/"
        ckpt_dir = "checkpoints/" + id_seed_dir
        ckpt = tf.train.get_checkpoint_state(
            ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists,
            # restore the parameters
            # and set epoch_n and i_iter
            model.g['saver'].restore(sess, ckpt.model_checkpoint_path)


        acc = model_eval(sess, x=model.g['images'], y=model.g['labels'],
                   X_test=dataset.test.images, Y_test=dataset.test.labels,
                   feed={model.g['train_flag']: False}, args=vars(params))

    print(acc)


if __name__ == '__main__':
    main()