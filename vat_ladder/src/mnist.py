"""
Functions for downloading and reading MNIST data.
Compatible with Python 3
"""

import gzip
import os
import urllib.request, urllib.parse, urllib.error

import numpy
from src.utils import DataSet, SemiDataSet, dense_to_one_hot

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
