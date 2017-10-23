# -----------------------------
# IMPORTS
# -----------------------------

import argparse
import numpy as np
import tensorflow as tf
from math import ceil

# -----------------------------
# DATASETS
# -----------------------------

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, preprocessed=False):
    if fake_data:
      self._num_examples = 10000
    elif preprocessed:
        self._num_examples = images.shape[0]
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in range(784)]
      fake_label = 0
      return [fake_image for _ in range(batch_size)], [
          fake_label for _ in range(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled, disjoint=False,
                 preprocessed=False):
        self._n_labeled = n_labeled
        self.disjoint = disjoint
        l_images, l_labels, u_images, u_labels = self.sample_balanced_labeled(images, labels, num_labeled=n_labeled)

        # Unlabeled DataSet
        if disjoint:
            self._unlabeled_ds = DataSet(u_images, u_labels, preprocessed=preprocessed)
        else:
            self._unlabeled_ds = DataSet(images, labels, preprocessed=preprocessed)

        # Labeled DataSet
        self._labeled_ds = DataSet(l_images, l_labels, preprocessed=preprocessed)

    def next_batch(self, batch_size, ul_batch_size=None):
        if ul_batch_size is None:
            ul_batch_size = batch_size
        unlabeled_images, _ = self.unlabeled_ds.next_batch(ul_batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = np.vstack([labeled_images, unlabeled_images])
        return images, labels

    def sample_balanced_labeled(self, images, labels, num_labeled):
        n_total, n_classes = labels.shape

        # First create a fully balanced set larger than desired
        nl_per_class = int(ceil(num_labeled / n_classes))
        # rng = self.rng
        idx = []

        for c in range(n_classes):
            c_idx = np.where(labels[:,c]==1)[0]
            idx.append(np.random.choice(c_idx, nl_per_class))
        l_idx = np.concatenate(idx)

        # Now sample uniformly without replacement from the larger set to get desired
        l_idx = np.random.choice(l_idx, size=num_labeled, replace=False)

        u_idx = np.setdiff1d(np.arange(n_total), l_idx, assume_unique=True)
        self.l_idx = l_idx
        self.u_idx = u_idx

        return images[l_idx], labels[l_idx], images[u_idx], labels[u_idx]



    @property
    def n_labeled(self):
        return self._n_labeled

    @property
    def num_examples(self):
        if self.disjoint:
            return self.num_labeled + self.num_unlabeled
        else:
            return self.num_unlabeled

    @property
    def labeled_ds(self):
        return self._labeled_ds

    @property
    def unlabeled_ds(self):
        return self._unlabeled_ds

    @property
    def num_labeled(self):
        return self._labeled_ds.num_examples

    @property
    def num_unlabeled(self):
        return self._unlabeled_ds.num_examples

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[(index_offset + labels_dense.ravel()).astype(int)] = 1
    return labels_one_hot

# -----------------------------
# PARAMETER PARSING
# -----------------------------

def parse_argstring(argstring, dtype=float, sep='-'):
    return list(map(dtype, argstring.split(sep)))

def get_cli_params():
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--test', action='store_true')
    # -------------------------
    # LOGGING
    # -------------------------
    add('--id', default='ladder')
    add('--logdir', default='results/logs/')
    add('--ckptdir', default='checkpoints/')
    add('--write_to', default=None)
    # description to print
    add('--description', default=None)

    # option to not save the model at all
    add('--do_not_save', action='store_true')
    add('--verbose', action='store_true')

    # -------------------------
    # DATA
    # -------------------------
    add('--dataset', default='mnist', choices=['mnist', 'svhn', 'cifar10'])
    add('--input_size', default=784, type=int)
    # -------------------------
    # EVALUATE
    # -------------------------
    add('--test_frequency_in_epochs', default=5, type=float)
    # validation
    add('--validation', default=0, nargs='?', const=1000, type=int)

    add('--tb', default=False, nargs='?', const='tb/')
    # -------------------------
    # TRAINING
    # -------------------------

    add('--which_gpu', default=0, type=int)
    add('--seed', default=1, type=int)

    add('--end_epoch', default=150, type=int)
    add('--num_labeled', default=100, type=int)
    add('--batch_size', default=100, type=int)
    add('--ul_batch_size', default=100, type=int)

    add('--initial_learning_rate', default=0.002, type=float)
    add('--decay_start', default=0.67, type=float)
    add('--lr_decay_frequency', default=5, type=int)

    add('--beta1', default=0.9, type=float) # first momentum coefficient
    add('--beta1_during_decay', default=0.9, type=float)

    # -------------------------
    # LADDER STRUCTURE
    # -------------------------
    # Specify encoder layers
    add('--encoder_layers',
                        default='1000-500-250-250-250-10')

    # Standard deviation of the Gaussian noise to inject at each level
    add('--corrupt_sd', default=0.3, type=float)

    # Standard deviation of the Gaussian noise to inject into clean images


    # Default RC cost corresponds to the gamma network
    add('--rc_weights', default='2000-20-0.2-0.2-0.2-0.2-0.2')

    # Batch norm decay weight mode
    add('--static_bn', default=0.99, type=float)

    # Use lrelu
    add('--lrelu_a', default=0.0, type=float)

    # Batch norm the top logits
    add('--top_bn', action='store_true')

    # -------------------------
    # VAT SETTINGS
    # -------------------------
    # vat params
    add('--epsilon', default='5.0')  # vary this instead of vat_weight
    add('--num_power_iters', default=1, type=int)
    add('--xi', default=1e-6, type=float, help="small constant for finite difference")
    add('--vadv_sd', default=0.0, type=float,
        help="noise to add at each layer of forward pass for stability")

    # -------------------------

    # VAL SETTINGS
    # -------------------------
    add('--model', default="c", choices=['c', 'clw', 'n', 'nlw', 'ladder',
                                         'supervised'])
    add('--decoder', default="full", choices=['gamma', 'full', 'none'])

    add('--measure_smoothness', action='store_true')
    add('--measure_vat', action='store_true', help='compute vat_cost but do '
                                                   'not use for optimisation')

    # -------------------------
    # CNN LADDER
    # -------------------------
    add('--cnn', action='store_true')

    # arguments for the cnn encoder/decoder
    add('--cnn_layer_types', default='c-c-c-max-c-c-c-max-c-c-c-avg-fc')
    add('--cnn_fan', default='3-96-96-96-96-192-192-192-192-192-192-192-192-10')
    add('--cnn_ksizes', default='3-3-3-3-3-3-3-3-3-1-1-0-0')
    add('--cnn_strides', default='1-1-1-2-1-1-1-2-1-1-1-0-0')
    add('--cnn_dims', default='32-32-32-32-16-16-16-16-8-8-8-8-1')

    # -------------------------
    # HYPEROPT
    # -------------------------
    parser.add_argument('--x0', default=None)
    parser.add_argument('--y0', default=None, type=float)
    parser.add_argument('--npi', default='1-2-3-4')

    # -------------------------
    # ADVERSARIAL ATTACKS
    # -------------------------
    add('--ord', default=2, choices=['inf', '1', '2'])

    params = parser.parse_args()

    return params


def enum_dict(list_):
    return dict(zip(range(len(list_)), list_))


def process_cli_params(params):
    # Specify base structure


    params.decay_start_epoch = int(params.decay_start * params.end_epoch)
    params.eval_batch_size = params.batch_size  # this should be redundant


    if params.cnn:
        params.cnn_layer_types = parse_argstring(params.cnn_layer_types,
                                                 dtype=str)
        params.cnn_fan = parse_argstring(params.cnn_fan, dtype=int)
        params.cnn_ksizes = parse_argstring(params.cnn_ksizes, dtype=int)
        params.cnn_strides = parse_argstring(params.cnn_strides, dtype=int)
        params.cnn_dims = parse_argstring(params.cnn_dims, dtype=int)
        params.encoder_layers = params.cnn_fan
        # params.rc_weights = enum_dict(([0] * (len(params.cnn_fan)-1)) + [float(
        #     params.rc_weights)])

        if params.dataset == "mnist":
            params.cnn_init_size = 28
            # params.cnn_fan[0] = 1
            params.input_size = 1
        elif params.dataset == "cifar10":
            params.cnn_init_size = 32
            # params.cnn_fan[0] = 3
            params.input_size = 3
        else:
            params.cnn_init_size = 28
            params.input_size = 1

    else:
        params.encoder_layers = parse_argstring(params.encoder_layers,
                                                dtype=int)
        if params.dataset == 'mnist':
            params.input_size = 784
        elif params.dataset == 'svhn':
            params.input_size = 768
        elif params.dataset == 'cifar10':
            params.input_size = 32 * 32
        else:
            params.input_size = 784

        params.encoder_layers = [params.input_size] + params.encoder_layers
    params.rc_weights = enum_dict(
        parse_argstring(params.rc_weights, dtype=float))

    params.num_layers = len(params.encoder_layers) - 1
    params.epsilon = enum_dict(parse_argstring(params.epsilon, dtype=float))
    # if params.model == 'vat':
    #     params.epsilon = params.epsilon[0]

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

def preprocess(placeholder, params):
    return tf.reshape(placeholder, shape=[
        -1, params.cnn_init_size, params.cnn_init_size, params.cnn_fan[0]
    ]) if params.cnn else placeholder


def get_batch_ops(batch_size):
    join = lambda l, u: tf.concat([l, u], 0)
    split_lu = lambda x: (labeled(x), unlabeled(x))
    labeled = lambda x: x[:batch_size] if x is not None else x
    unlabeled = lambda x: x[batch_size:] if x is not None else x
    return join, split_lu, labeled, unlabeled


