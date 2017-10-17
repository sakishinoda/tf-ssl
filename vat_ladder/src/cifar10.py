from src.utils import DataSet, SemiDataSet, dense_to_one_hot

import os
import sys
import tarfile

import numpy as np
from scipy import linalg
import glob
import pickle
import urllib.request, urllib.parse, urllib.error

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 50000
NUM_EXAMPLES_TEST = 10000



def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

def maybe_download_and_extract(data_dir, downsample=False):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    # Training set
    print("Loading training data...")
    train_images = np.zeros((NUM_EXAMPLES_TRAIN, 3 * 32 * 32), dtype=np.float32)
    train_labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob(data_dir + '/cifar-10-batches-py/data_batch*'))):
        batch = unpickle(data_fn)
        train_images[i * 10000:(i + 1) * 10000] = batch['data']
        train_labels.extend(batch['labels'])
    train_images = (train_images - 127.5) / 255.
    train_labels = np.asarray(train_labels, dtype=np.int64)

    rand_ix = np.random.permutation(NUM_EXAMPLES_TRAIN)
    train_images = train_images[rand_ix]
    train_labels = train_labels[rand_ix]

    print("Loading test data...")
    test = unpickle(data_dir + '/cifar-10-batches-py/test_batch')
    test_images = test['data'].astype(np.float32)
    test_images = (test_images - 127.5) / 255.
    test_labels = np.asarray(test['labels'], dtype=np.int64)
    assert all(test_labels < 10), "Labels not in [0, 9]"

    print("Apply ZCA whitening")
    components, mean, train_images = ZCA(train_images)
    np.save('{}/components'.format(data_dir), components)
    np.save('{}/mean'.format(data_dir), mean)
    test_images = np.dot(test_images - mean, components.T)

    train_images = train_images.reshape(
        (NUM_EXAMPLES_TRAIN, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TRAIN, -1))
    test_images = test_images.reshape(
        (NUM_EXAMPLES_TEST, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TEST, -1))

    np.save('{}/train_images'.format(data_dir), train_images)
    np.save('{}/train_labels'.format(data_dir), train_labels)
    np.save('{}/test_images'.format(data_dir), test_images)
    np.save('{}/test_labels'.format(data_dir), test_labels)

    # return (train_images, train_labels), (test_images, test_labels)



def load_cifar10(data_dir):
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

    train_images, train_labels, test_images, test_labels = load_cifar10(
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
    maybe_download_and_extract('../../data/cifar10/')