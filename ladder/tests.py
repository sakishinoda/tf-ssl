
import IPython
import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))
from src import feed
from tensorflow.examples.tutorials.mnist import input_data
from src import utils



def test_data_balancing():
    # Test mnist data balancing z
    mnist = input_data.read_data_sets(sys.path[0]+'/../data/mnist/', one_hot=True)
    sf = feed.Balanced(mnist.validation.images, mnist.validation.labels, 100, None)
    IPython.embed()


def test_seeds():
    params = utils.get_cli_params()

