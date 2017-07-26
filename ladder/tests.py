
import IPython
import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))
from src import feed, utils, ladder
from tensorflow.examples.tutorials.mnist import input_data
import csv
import tensorflow as tf


def test_data_balancing():
    # Test mnist data balancing z
    mnist = input_data.read_data_sets(sys.path[0]+'/../data/mnist/', one_hot=True)
    sf = feed.Balanced(mnist.validation.images, mnist.validation.labels, 100, None)
    IPython.embed()


def test_similarity(file1, file2):
    """Test line-by-line similarity of two tsv files """
    with open(file1, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        data1 = list(reader)

    with open(file2, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        data2 = list(reader)

    sim = []
    for i, l in enumerate(data1):
        sim.append(1 if (data2[i]==l) else 0)

        print(i, sim[-1])
        # if sim[-1]==0:
        #     print(">>>>> MISMATCH <<<<<")
        #     print(l)
        #     print(data2[i])
        #     print("<<<<< END >>>>>")

    return sim


def test_seeds():
    """
    Small test to check reproducibility with seeds.

    """
    params = utils.process_cli_params(utils.get_cli_params())

    # Alter from default to simplify
    params.decay_start_epoch = 1
    params.end_epoch = 2
    params.train_flag = True
    params.gamma_flag = True
    params.num_labeled = 100

    # params.seed = 1

    for run in [1, 2]:
        tf.reset_default_graph()
        # run = 2
        params.seed = run
        params.write_to = "tests/repr_test_" + str(run)
        ladder.main(params)

    sim = test_similarity("tests/repr_test_1", "tests/repr_test_2")
    print(sum(sim)/len(sim))






def test_fully_supervised():
    """
    Test to check that the fully supervised case (i.e. no unsupervised examples)
    works for e.g. all of MNIST, 100 labels, etc.

    Easiest way is to set the num_labeled to desired, then all rc_weights to
    zero. This is not very efficient, but allows the code to be used as is.


    """

    params = utils.process_cli_params(utils.get_cli_params())

    # Alter from default to simplify
    params.decay_start_epoch = 1
    params.end_epoch = 2
    params.train_flag = True
    params.gamma_flag = True
    params.num_labeled = 100

    params.seed = 1
    params.rc_weights = [0 for x in params.rc_weights]

    tf.reset_default_graph()
    params.write_to = "repr_test_"


if __name__ == '__main__':
    test_seeds()