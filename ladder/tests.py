
import IPython
import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))
from src import feed, utils
from src import ladder_network as ldr
from tensorflow.examples.tutorials.mnist import input_data
import csv
import tensorflow as tf


def test_data_balancing():
    # Test mnist data balancing z
    mnist = input_data.read_data_sets(sys.path[0]+'/../data/mnist/', one_hot=True)
    sf = feed.Balanced(mnist.validation.images, mnist.validation.labels, 100, None)
    IPython.embed()


def get_testing_mode_params(gamma=True, num_labeled=100, seed=1):
    params = utils.process_cli_params(utils.get_cli_params())

    # Alter from default to simplify
    params.decay_start_epoch = 1
    params.end_epoch = 2
    params.train_flag = True
    params.gamma_flag = gamma
    params.num_labeled = num_labeled
    params.seed = seed

    return params


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


def test_repr():
    """
    Small test to check reproducibility with seeds.

    Results
    -------
        100% consistency over 2 epochs using same seed.
        31% consistency over 2 epochs using seeds 1 and 2.

    """
    params = get_testing_mode_params()
    # params.seed = 1

    for run in [1, 2]:
        tf.reset_default_graph()
        # run = 2
        params.seed = run
        params.id = "repr_" + str(run)
        params.write_to = "tests/" + params.id
        ldr.main(params)

    sim = test_similarity("tests/" + "repr_" + str(1), "tests/" + "repr_" +
                          str(2))
    print(sum(sim)/len(sim))


def test_fully_supervised():
    """
    Test to check that the fully supervised case (i.e. no unsupervised examples)
    works for e.g. all of MNIST, 100 labels, etc.

    Easiest way is to set the num_labeled to desired, then all rc_weights to
    zero. This is not very efficient, but allows the code to be used as is.

    Results
    -------
    Overwrote when using labeled epochs. Turns out labeled epochs are shorter
    than print interval for 100, 1000 labels.
    Using unlabeled epochs, found expected patterns of overfitting.

    To Do
    -----
    This suggests an experiment that can be written up showing the
    overfitting tendency on 100 and how ssl effectively regularises using
    information unsupervised. Could do a comparison with alternative
    regularisation techniques or see if anyone else has tried this. As in,
    someone must have tried it -- but did they do it in a form where I could
    copy it into my writeup?

    """

    params = get_testing_mode_params()

    params.seed = 1
    params.rc_weights = [0 for x in params.rc_weights]
    params.use_labeled_epochs = True

    for num_labeled in [100, 1000, 55000]:
        tf.reset_default_graph()
        params.num_labeled = num_labeled
        params.id = "fully_sup_" + str(num_labeled)
        params.write_to = "tests/" + params.id
        ldr.main(params)


def test_if_zero_rc_is_dummy():
    """Weights for each reconstruction cost are specified using a dictionary
    with keys from 0 to L supposed to match the layer keys, but quite
    possibly the very first entry (0th layer) RC isn't actually used."""

    params = get_testing_mode_params(gamma=False)

    # Use only the RC, not supervised
    params.sc_weight = 0

    # Gamma setting
    rc_weights = [0, 0, 0, 0, 0, 0, 1]
    params.rc_weights = dict(zip(range(len(rc_weights)), rc_weights))
    tf.reset_default_graph()

    params.id = "rc_zero_base"
    params.write_to = "tests/" + params.id
    ldr.main(params)

    # Gamma setting
    rc_weights = [1, 0, 0, 0, 0, 0, 1]
    params.rc_weights = dict(zip(range(len(rc_weights)), rc_weights))
    tf.reset_default_graph()

    params.id = "rc_zero_one"
    params.write_to = "tests/" + params.id
    ldr.main(params)

    sim = test_similarity("tests/rc_zero_base", "tests/rc_zero_one")
    print(sum(sim) / len(sim))



def test_identify_layer_numbers():
    params = get_testing_mode_params(gamma=False)
    ladder = ldr.Ladder(params)

    print("Noisy encoder activations")
    for k,v in ladder.noisy.z.items():
        print(k, v)

    print("Decoder costs")
    for k,v in ladder.decoder.rc_cost.items():
        print(k, v)



    print("Reconstruction cost weights")
    for k, v in params.rc_weights.items():
        print(k, v)




def test_only_unsupervised():
    """
    Test unsupervised reconstruction costs and weight effects on overall loss.

    Result
    ------
    Except for difference in timing, results are identical, meaning that
    there is no zero layer.

    Is there supposed to be a reconstruction cost associated with the final
    output label? Well, the final output label is an argmax. So as long as
    there is a cost associated with the ten probabilities layer, all good.

    """
    params = get_testing_mode_params()

    params.seed = 1
    params.sc_weight = 0

    # Test sequences for weights
    # Key:      0 1 2 3 4 5 6
    # Default:  0-0-0-0-0-0-1
    rc_weights = [0, 0, 0, 0, 0, 0, 1]
    rc_weights = [1, 0, 0, 0, 0, 0, 1]

    params.rc_weights = dict(zip(range(len(rc_weights)), rc_weights))

    for num_labeled in [100, 1000, 55000]:
        tf.reset_default_graph()
        params.num_labeled = num_labeled
        params.id = "fully_sup_" + str(num_labeled)
        params.write_to = "tests/" + params.id
        ldr.main(params)

if __name__ == '__main__':
    test_identify_layer_numbers()
    # test_if_zero_rc_is_dummy()
