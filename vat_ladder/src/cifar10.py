from src.utils import DataSet, SemiDataSet, dense_to_one_hot

import os
import sys
import tarfile

import numpy as np
import scipy
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


def cnorm(data, scale=55, epsilon=1e-8):
    """
    Global contrast normalisation function.
    First convert to float32.
    Centre and scale all pixels in an image by that image mean and std.

    :param data:
    :param scale:
    :return:
    """
    assert len(data.shape) == 2, "Data not flattened"
    scale = np.float32(scale)
    epsilon = np.float32(epsilon)

    # Convert to float32
    data = np.require(data, dtype=np.float32)  # convert from uint8 to float

    # Centre
    data -= data.mean(axis=1)[:, np.newaxis]

    # Scale
    norms = np.sqrt(np.sum(data ** 2, axis=1)) / scale
    norms[norms < epsilon] = np.float32(1.)
    data /= norms[:, np.newaxis]

    return data



def ZCA(data, n_components=3072, filter_bias=0.1):

    # Check data already flattened
    assert data.shape[1] == n_components

    assert n_components == np.product(data.shape[1:]), 'ZCA whitening components should be {} for convolutional data'.format(np.product(data.shape[1:]))

    mean = np.mean(data, axis=0)
    bias = filter_bias * scipy.sparse.identity(data.shape[1], 'float32')
    cov = np.cov(data, rowvar=0, bias=1) + bias
    eigs, eigv = scipy.linalg.eigh(cov)

    assert not np.isnan(eigs).any()
    assert not np.isnan(eigv).any()
    assert eigs.min() > 0

    eigs = eigs[-n_components:]
    eigv = eigv[:, -n_components:]

    sqrt_eigs = np.sqrt(eigs)
    components = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
    whiten = np.dot(data - mean, components)
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

    # =============
    # Training set
    print("Loading training data...")
    train_images = np.zeros((NUM_EXAMPLES_TRAIN, 3 * 32 * 32), dtype=np.float32)
    train_labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob(data_dir + '/cifar-10-batches-py/data_batch*'))):
        batch = unpickle(data_fn)
        train_images[i * 10000:(i + 1) * 10000] = batch['data']
        train_labels.extend(batch['labels'])

    # Contrast normalization (i.e. normalize all pixels in each image by the
    # mean and standard deviation of that image)
    # Data is already flattened
    # train_images = (train_images - 127.5) / 255.
    print("Contrast-normalizing training data...")
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_images = cnorm(train_images)

    # Random ordering of training examples
    rand_ix = np.random.permutation(NUM_EXAMPLES_TRAIN)
    train_images = train_images[rand_ix]
    train_labels = train_labels[rand_ix]

    # =============
    # Testing set
    print("Loading test data...")
    test = unpickle(data_dir + '/cifar-10-batches-py/test_batch')
    test_images = test['data'].astype(np.float32)
    test_images = cnorm(test_images)

    print("Contrast-normalizing testing data...")
    # Convert to [0,1] float
    # test_images = (test_images - 127.5) / 255.
    test_labels = np.asarray(test['labels'], dtype=np.int64)
    assert all(test_labels < 10), "Labels not in [0, 9]"

    # =============
    # ZCA on flattened data
    print("Applying ZCA whitening...")
    components, mean, train_images = ZCA(train_images)
    np.save('{}/components'.format(data_dir), components)
    np.save('{}/mean'.format(data_dir), mean)
    test_images = np.dot(test_images - mean, components)

    # =============
    # Unflatten data
    train_images = train_images.reshape(
        (NUM_EXAMPLES_TRAIN, 3, 32, 32)).transpose((0, 2, 3, 1))
    #.reshape((NUM_EXAMPLES_TRAIN, -1))
    test_images = test_images.reshape(
        (NUM_EXAMPLES_TEST, 3, 32, 32)).transpose((0, 2, 3, 1))
    #.reshape((NUM_EXAMPLES_TEST, -1))tn

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


def read_data_sets(train_dir, n_labeled=4000, fake_data=False,
                   one_hot=True, verbose=False, validation_size=0,
                   disjoint=False):

    class DataSets(object):
        pass

    data_sets = DataSets()

    VALIDATION_SIZE = validation_size


    for x in ['train_images.npy',
              'train_labels.npy',
              'test_images.npy',
              'test_labels.npy']:
        filepath = os.path.join(train_dir, x)

        if not os.path.exists(filepath):
            maybe_download_and_extract(data_dir=train_dir)
            break
        else:
            print(filepath, "found")

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
