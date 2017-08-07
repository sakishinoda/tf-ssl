# -----------------------------
# IMPORTS
# -----------------------------
from tensorflow.contrib import layers as layers
import argparse
import numpy as np
import tensorflow as tf
import math
# -----------------------------
# PARAMETER PARSING
# -----------------------------

def parse_argstring(argstring, dtype=float, sep='-'):
    return list(map(dtype, argstring.split(sep)))

def get_cli_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='ladder')
    # parser.add_argument('--train_flag', action='store_true')
    parser.add_argument('--decay_start_epoch', default=100, type=int)
    parser.add_argument('--end_epoch', default=150, type=int)
    parser.add_argument('--test_frequency_in_epochs', default=1, type=int)
    # parser.add_argument('--print_interval', default=50, type=int)
    # parser.add_argument('--save_epochs', default=None, type=float)
    parser.add_argument('--num_labeled', default=100, type=int)

    parser.add_argument('--batch_size', default=100, type=int)
    # parser.add_argument('--unlabeled_batch_size', default=250, type=int)

    parser.add_argument('--initial_learning_rate', default=0.002, type=float)

    # parser.add_argument('--gamma_flag', action='store_true')

    # Specify encoder layers
    parser.add_argument('--encoder_layers',
                        default='784-1000-500-250-250-250-10')

    # Weight to apply to supervised cost in total loss
    # parser.add_argument('--sc_weight', default=1, type=float)

    # Standard deviation of the Gaussian noise to inject at each level
    parser.add_argument('--encoder_noise_sd', default=0.3, type=float)

    # Default RC cost corresponds to the gamma network
    parser.add_argument('--rc_weights', default='2000-20-0.2-0.2-0.2-0.2-0.2')

    # Specify form of combinator (A)MLP
    # parser.add_argument('--combinator_layers', default='4-1')
    parser.add_argument('--combinator_sd', default=0.025, type=float)

    parser.add_argument('--which_gpu', default=0, type=int)
    parser.add_argument('--write_to', default=None)
    parser.add_argument('--seed', default=1, type=int)

    # by default use the unlabeled batch epochs
    # parser.add_argument('--use_labeled_epochs', action='store_true')

    # only used if train_flag is false
    parser.add_argument('--train_step', default=None, type=int)
    parser.add_argument('--verbose', action='store_true') # for testing

    # option to not save the model at all
    parser.add_argument('--do_not_save', action='store_true')

    # vat params
    parser.add_argument('--epsilon', default = 8.0, type=float)
    parser.add_argument('--num_power_iterations', default=1, type=int)
    parser.add_argument('--xi', default=1e-6, type=float)


    # weight of vat cost
    parser.add_argument('--vat_weight', default=0, type=float)
    parser.add_argument('--vat_rc', action='store_true')
    parser.add_argument('--corrupt', default='gauss', choices=['gauss',
                                                               'vatgauss',
                                                               'vat'])

    # weight of entropy minimisation cost
    parser.add_argument('--ent_weight', default=0, type=float)

    # description to print
    parser.add_argument('--description', default=None)

    parser.add_argument('--cnn', action='store_true')
    # arguments for the cnn encoder/decoder
    parser.add_argument('--cnn_init_size', default=32, type=int)

    parser.add_argument('--keep_prob_hidden', default=0.5, type=float)
    parser.add_argument('--lrelu_a', default=0.1, type=float)
    parser.add_argument('--top_bn', action='store_true')
    parser.add_argument('--bn_stats_decay_factor', default=0.99, type=float)

    params = parser.parse_args()
    # params.write_to = 'logs/' + params.id + '.results' if params.write_to is None else params.write_to

    return params

def process_cli_params(params):
    # Specify base structure
    encoder_layers = parse_argstring(params.encoder_layers, dtype=int)
    rc_weights = parse_argstring(params.rc_weights, dtype=float)
    rc_weights = dict(zip(range(len(rc_weights)), rc_weights))
    params.encoder_layers = encoder_layers
    params.rc_weights = rc_weights

    if params.cnn:
        params.cnn_layer_types = ('c', 'c', 'c', 'max', 'c', 'c', 'c', 'max',
                                'c', 'c',
                       'c', 'avg', 'fc')
        params.cnn_fan = (3, 96, 96, 96, 96, 192, 192, 192, 192, 192, 192, 192,
                      192, 10)
        params.cnn_ksizes = (3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, None, None)
        params.cnn_strides = (1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, None, None)
        params.num_layers = len(params.cnn_fan) - 1
        # assert len(params.rc_weights) == len(params.cnn_fan) -1


    else:
        params.num_layers = len(params.encoder_layers) - 1

    # NUM_EPOCHS = params.end_epoch
    # NUM_LABELED = params.num_labeled

    return params

def count_trainable_params():
    trainables = tf.trainable_variables()
    return np.sum([np.prod(var.get_shape()) for var in trainables])

def order_param_settings(params):
    param_dict = vars(params)
    param_list = []
    for k in sorted(param_dict.keys()):
        param_list.append(str(k) + ": " + str(param_dict[k]))

    return param_list


# -----------------------------
# MODEL BUILDING
# -----------------------------

def fclayer(input,
            size_out,
            wts_init=layers.xavier_initializer(),
            bias_init=tf.truncated_normal_initializer(stddev=1e-6),
            reuse=None,
            scope=None,
            activation=None):
    return layers.fully_connected(
        inputs=input,
        num_outputs=size_out,
        activation_fn=activation,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=wts_init,
        weights_regularizer=None,
        biases_initializer=bias_init,
        biases_regularizer=None,
        reuse=reuse,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=scope
    )


def bias_init(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)

def wts_init(shape, name):
    # effectively a Xavier initializer
    return tf.Variable(tf.random_normal(shape), name=name) / \
           math.sqrt(shape[0])


