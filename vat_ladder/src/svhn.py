from src.utils import DataSet, SemiDataSet, dense_to_one_hot
from scipy.io import loadmat
from scipy.misc import toimage, imresize
import numpy as np
import os
import sys
from scipy import linalg
import urllib.request, urllib.parse, urllib.error

# from dataset_utils import *

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032


def get_downsampled_batch(batch):
    new_batch = np.zeros((batch.shape[0], int(batch.shape[1] * 0.5 *
                          batch.shape[2] * 0.5 * batch.shape[3])),
                         dtype=batch.dtype)

    for i in range(batch.shape[0]):
        im = toimage(np.squeeze(batch[i]), channel_axis=2)
        im = imresize(im, size=0.5)
        new_batch[i, :] = im.reshape([1, -1])

    return new_batch


def maybe_download_and_extract(data_dir, downsample=False):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filepath_train_mat = os.path.join(data_dir, 'train_32x32.mat')
    filepath_test_mat = os.path.join(data_dir, 'test_32x32.mat')

    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

    # Training set
    print('\n')
    print("Loading training data...")
    print("Preprocessing training data...")
    train_data = loadmat(filepath_train_mat)
    train_x = (-127.5 + train_data['X']) / 255.
    train_x = train_x.transpose((3, 0, 1, 2))
    if downsample:
        train_x = get_downsampled_batch(train_x)
    else:
        train_x = train_x.reshape([train_x.shape[0], -1])
    train_y = train_data['y'].flatten().astype(np.int32)
    train_y[train_y == 10] = 0

    # Test set
    print("Loading test data...")
    test_data = loadmat(filepath_test_mat)
    test_x = (-127.5 + test_data['X']) / 255.  # centering
    test_x = test_x.transpose((3, 0, 1, 2))
    if downsample:
        test_x = get_downsampled_batch(test_x)
    else:
        test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = test_data['y'].flatten().astype(np.int32)
    test_y[test_y == 10] = 0

    np.save('{}/train_images'.format(data_dir), train_x)
    np.save('{}/train_labels'.format(data_dir), train_y)
    np.save('{}/test_images'.format(data_dir), test_x)
    np.save('{}/test_labels'.format(data_dir), test_y)


def load_svhn(data_dir):
    train_images = np.load('{}/train_images.npy'.format(data_dir)).astype(np.float32)
    train_labels = np.load('{}/train_labels.npy'.format(data_dir)).astype(np.float32)
    test_images = np.load('{}/test_images.npy'.format(data_dir)).astype(np.float32)
    test_labels = np.load('{}/test_labels.npy'.format(data_dir)).astype(np.float32)
    return train_images, train_labels, test_images, test_labels


def read_data_sets(train_dir, n_labeled=1000, fake_data=False,
                   one_hot=True, verbose=False, validation_size=0,
                   disjoint=False, downsample=True, download_and_extract=True):

    class DataSets(object):
        pass

    data_sets = DataSets()

    VALIDATION_SIZE = validation_size

    if download_and_extract:
        maybe_download_and_extract(data_dir=train_dir, downsample=downsample)

    train_images, train_labels, test_images, test_labels = load_svhn(
        train_dir)

    if one_hot:
        train_labels = dense_to_one_hot(train_labels, 10)
        test_labels = dense_to_one_hot(test_labels, 10)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = SemiDataSet(train_images, train_labels, n_labeled,
                                  disjoint=disjoint, preprocessed=True)
    data_sets.validation = DataSet(validation_images, validation_labels,
                                   preprocessed=True)
    data_sets.test = DataSet(test_images, test_labels, preprocessed=True)

    return data_sets


if __name__ == "__main__":
    read_data_sets('../../data/svhn/')