import tensorflow as tf
import os
from src.lva import build_graph, build_vat_graph
from src.train import evaluate_metric
from src.mnist import read_data_sets
# import numpy as np
from src.utils import process_cli_params, get_cli_params, parse_argstring
from skopt import gp_minimize, dump
from tqdm import tqdm
import argparse

class Hyperopt(object):
    def __init__(self):
        # Parse command line and default parameters
        self.params = process_cli_params(get_cli_params())

        # for k in sorted(self.params_dict.keys()):
        #     print(k, self.params_dict[k])


    def objective(self, x):

        p = self.convert_dims_to_params(x)
        if p.verbose:
            print("----------------------------------------")
            print("----------------------------------------")


        tf.reset_default_graph()

        # -----------------------------
        # Set GPU device to use
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(p.which_gpu)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Set seeds
        # np.random.seed(p.seed)
        # tf.set_random_seed(p.seed)

        # Load data
        dataset = read_data_sets("MNIST_data",
                                 n_labeled=p.num_labeled,
                                 validation_size=p.validation,
                                 one_hot=True,
                                 disjoint=False)
        num_examples = dataset.train.num_examples
        if p.validation > 0:
            dataset.test = dataset.validation
        iter_per_epoch = (num_examples // p.batch_size)
        num_iter = iter_per_epoch * p.end_epoch

        # Build graph
        if p.model == 'vat':
            g, m, trainable_parameters = build_vat_graph(p)
        else:
            g, m, trainable_parameters = build_graph(p)

        val_errs = []
        error = tf.constant(100.0) - m['acc']

        if p.verbose:
            print("=== Starting Session ===")
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if p.verbose:
                print("=== Training ===")

            for i in tqdm(range(num_iter)):

                images, labels = dataset.train.next_batch(p.batch_size,
                                                          p.ul_batch_size)
                _ = sess.run(
                    [g['train_step']],
                    feed_dict={g['images']: images,
                               g['labels']: labels,
                               g['train_flag']: True})

                if (i > 1) and ((i + 1) % int(p.test_frequency_in_epochs *
                                                  iter_per_epoch) == 0):
                    val_err = evaluate_metric(dataset.validation, sess, error, graph=g, params=p)
                    # print((i+1)//iter_per_epoch, val_err, sep='\t')
                    val_errs.append(val_err)

        val_err = min(val_errs)

        return val_err

    def get_dims(self):
        if self.params.model == "vat":
            dims = [
                (0.01, 10.0, 'log-uniform')  # 0: eps
            ]

            x0 = [5.0]

        elif self.params.model == "ladder":
            dims = [
                # rc_weights
                (500., 5000, 'log-uniform'),  # 0 rc_0
                (5.00, 50., 'log-uniform'),  # 1 rc_1
                (0.01, 1.0, 'log-uniform')
            ]
            x0 = [1000, 10, 0.1]

        else:
            dims = [
                # rc_weights
                (500., 5000, 'log-uniform'),  # 0 rc_0
                (5.00, 50., 'log-uniform'),  # 1 rc_1
                (0.01, 1.0, 'log-uniform'),  # 2 rc_2:6
                (0.01, 10.0, 'log-uniform')  # 3: eps_0
            ]
            x0 = [1000, 10, 0.1, 1.0]

            if self.params.model == 'clw' or self.params.model == 'nlw':
                dims += [
                    (1e-3, 0.5, 'log-uniform'),  # 4 eps_1
                    (1e-5, 0.1, 'log-uniform'),  # 5 eps_2
                ]
                x0 += [0.1, 0.001]

        if self.params.x0 is not None:
            x0 = parse_argstring(self.params.x0, dtype=float, sep=',')

        y0 = self.params.y0

        return dims, x0, y0


    def convert_dims_to_params(self, x):

        # -------------------------
        # Optimize
        if self.params.model == "vat":
            self.params.epsilon = {0: x[0]}

        else:
            self.params.rc_weights = {0: x[0], 1: x[1],
                                      2: x[2], 3: x[2],
                                      4: x[2], 5: x[2],
                                      6: x[2]
                                      }

            if self.params.model == 'clw' or self.params.model == 'nlw':
                self.params.epsilon = {0: x[3], 1: x[4],
                                       2: x[5], 3: x[5],
                                       4: x[5], 5: x[5],
                                       6: x[5]
                                       }
            elif self.params.model == 'n' or self.params.model == 'c':
                self.params.epsilon = {0: x[3]}

        print("x:", x)
        return self.params



from statistics import mean, stdev

class HyperoptNPI(Hyperopt):
    def convert_dims_to_params(self, x):
        self.params.seed = x[0]
        self.params.num_power_iters = x[1]

        return self.params


def test_num_power_iters():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # filter out info, warnings

    hyperopt = HyperoptNPI()
    val_errs = {}

    npi_to_test = parse_argstring(hyperopt.params.npi, dtype=int)
    for npi in npi_to_test:
        print('====================')

        val_errs[npi] = []
        for seed in [1, 11, 111]:
            this_val_err = hyperopt.objective([seed, npi])
            val_errs[npi].append(this_val_err)
            print(npi, seed, this_val_err)
        print('--------------------')

    print('====================')
    for npi in npi_to_test:
        print(npi, mean(val_errs[npi]), stdev(val_errs[npi]))





def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # filter out info, warnings

    hyperopt = Hyperopt()
    dims, x0, y0 = hyperopt.get_dims()

    print("=== Beginning Search ===")
    res = gp_minimize(hyperopt.objective, dims, n_calls=11, x0=x0,
                      y0=y0, verbose=True)
    print(res.fun, ":", *res.x)

    dump_path = hyperopt.params.logdir + hyperopt.params.id + '.res'
    if not os.path.exists(hyperopt.params.logdir):
        os.makedirs(hyperopt.params.logdir)
    dump(res, dump_path)



if __name__ == '__main__':
    # tune_single_parameter()
    main()

