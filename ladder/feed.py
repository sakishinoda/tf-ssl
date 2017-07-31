# from tensorflow.python.framework import random_seed
# https://www.tensorflow.org/versions/r0.12/api_docs/python/constant_op/random_tensors#set_random_seed

import numpy as np
from math import ceil

class SemiFeed(object):
    def __init__(self, all_images, all_labels, num_labeled, seed=None):
        # seed1, seed2 = random_seed.get_seed(seed)
        # self.seeds = (seed, seed1, seed2)
        # If op level seed is not set, use whatever graph level seed is returned
        # self.rng = np.random.RandomState(seed1 if seed is None else seed2)
        images, labels, u_images, u_labels = self.sample_labeled(
            all_images, all_labels, num_labeled)
        self.labeled = Dataset(images, labels, seed)
        self.unlabeled = Dataset(u_images, u_labels, seed)
        # np.random.seed(seed)
        # self.seeds = seed

    def next_batches(self, l_batch_size, u_batch_size, shuffle=True):
        l_images, l_labels = self.labeled.next_batch(l_batch_size, shuffle)
        u_images, u_labels = self.unlabeled.next_batch(u_batch_size, shuffle)
        return l_images, l_labels, u_images, u_labels

    def next_batch(self, l_batch_size, u_batch_size, shuffle=True):
        li, ll, ui, ul = self.next_batches(l_batch_size, u_batch_size, shuffle)
        # return np.concatenate([li, ui]), np.concatenate([ll, ul])
        return np.concatenate([li,ui]), ll

    def sample_labeled(self, images, labels, num_labeled):

        perm = np.arange(images.shape[0])
        # self.rng.shuffle(perm)
        np.random.shuffle(perm)

        return images[perm[:num_labeled]], labels[perm[:num_labeled]], \
               images[perm[num_labeled:]], labels[perm[num_labeled:]]



class Dataset(object):
    def __init__(self, images, labels, seed=None):
        # seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        # self.rng = np.random.RandomState(seed1 if seed is None else seed2)
        self.images, self.labels= images, labels
        assert images.shape[0] == labels.shape[0]
        self.num_examples = self.images.shape[0]
        self.epochs_completed = 0
        self._index_in_epoch = 0
        assert self.images.shape[0] == self.num_examples


    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set.
        Adapted https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
        """

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle()

        # Go to the next epoch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            images_rest_part = self.images[start:self.num_examples]
            labels_rest_part = self.labels[start:self.num_examples]

            # Shuffle the data for next epoch
            if shuffle:
                self.shuffle()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.images[start:end], self.labels[start:end]

    def shuffle(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        # self.rng.shuffle()
        self.images = self.images[perm]
        self.labels = self.labels[perm]



class Balanced(SemiFeed):

    def sample_labeled(self, all_images, all_labels, num_labeled):

        assert num_labeled <= all_images.shape[0]

        images, labels, u_images, u_labels = self.sample_balanced_labeled(
            all_images,
            all_labels,
            num_labeled
        )

        return images, labels, u_images, u_labels

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

        return images[l_idx], labels[l_idx], images[u_idx], labels[u_idx]






