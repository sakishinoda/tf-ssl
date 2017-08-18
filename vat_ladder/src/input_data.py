"""
Functions for downloading and reading MNIST data.
Compatible with Python 3
"""

import gzip
import os
import urllib.request, urllib.parse, urllib.error

from math import ceil
import numpy

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory, verbose=False):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    if verbose:
        print(('Succesfully downloaded', filename, statinfo.st_size, 'bytes.'))
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename, verbose=False):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  if verbose:
      print(('Extracting', filename))
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False, verbose=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  if verbose:
      print(('Extracting', filename))
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
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
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled, disjoint=False):
        self._n_labeled = n_labeled
        self.disjoint = disjoint
        l_images, l_labels, u_images, u_labels = self.sample_balanced_labeled(images, labels, num_labeled=n_labeled)

        # Unlabeled DataSet
        if disjoint:
            self._unlabeled_ds = DataSet(u_images, u_labels)
        else:
            self._unlabeled_ds = DataSet(images, labels)

        # Labeled DataSet
        self._labeled_ds = DataSet(l_images, l_labels)

    def next_batch(self, batch_size, ul_batch_size=None):
        if ul_batch_size is None:
            ul_batch_size = batch_size
        unlabeled_images, _ = self.unlabeled_ds.next_batch(ul_batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels

    def sample_balanced_labeled(self, images, labels, num_labeled):
        n_total, n_classes = labels.shape

        # First create a fully balanced set larger than desired
        nl_per_class = int(ceil(num_labeled / n_classes))
        # rng = self.rng
        idx = []

        for c in range(n_classes):
            c_idx = numpy.where(labels[:,c]==1)[0]
            idx.append(numpy.random.choice(c_idx, nl_per_class))
        l_idx = numpy.concatenate(idx)

        # Now sample uniformly without replacement from the larger set to get desired
        l_idx = numpy.random.choice(l_idx, size=num_labeled, replace=False)

        u_idx = numpy.setdiff1d(numpy.arange(n_total), l_idx, assume_unique=True)

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



def read_data_sets(train_dir, n_labeled = 100, fake_data=False,
                   one_hot=False, verbose=False, validation_size=0,
                   disjoint=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = validation_size

  local_file = maybe_download(TRAIN_IMAGES, train_dir, verbose=verbose)
  train_images = extract_images(local_file, verbose=verbose)

  local_file = maybe_download(TRAIN_LABELS, train_dir, verbose=verbose)
  train_labels = extract_labels(local_file, one_hot=one_hot, verbose=verbose)

  local_file = maybe_download(TEST_IMAGES, train_dir, verbose=verbose)
  test_images = extract_images(local_file, verbose=verbose)

  local_file = maybe_download(TEST_LABELS, train_dir, verbose=verbose)
  test_labels = extract_labels(local_file, one_hot=one_hot, verbose=verbose)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = SemiDataSet(train_images, train_labels, n_labeled,
                                disjoint=disjoint)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets
