from cleverhans.model import Model as CleverHansModel
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from src.lva import build_graph_from_inputs, build_graph
from src.utils import get_cli_params, process_params, parse_argstring
from src.mnist import read_data_sets
import tensorflow as tf
import pickle
import os


class MyModel(CleverHansModel):
    def __init__(self, params):
        super(MyModel, self).__init__()
        self.params = params
        self.scope = tf.get_variable_scope()
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            self.g, self.m, _ = build_graph(params, is_training=False)
            self.outputs = self.g['labels']
            self.train_flag = self.g['train_flag']


    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        # return self.get_layer(x, 'logits')
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):

            g, m, _ = build_graph_from_inputs(x,
                                              self.outputs,
                                              self.train_flag,
                                              self.params, is_training=False)
            return g['logits']

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        with tf.variable_scope(self.scope, reuse=True):
            g, m, _ = build_graph_from_inputs(x,
                                              self.outputs,
                                              self.train_flag,
                                              self.params,
                                              is_training=False)

            return g['softmax']

def extract_model_name_from_path(path):
    model = (path.split('full_')[1]).split('/')[0].replace('_labeled-', '')
    seed = path.split('seed-')[1].split('/')[0]
    return model + '_seed-' + seed

def test_aer_on_normal_and_adv(p):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    results = {}
    tf.reset_default_graph()

    model = MyModel(p)
    dataset = read_data_sets("MNIST_data",
                             n_labeled=p.num_labeled,
                             validation_size=p.validation,
                             one_hot=True,
                             disjoint=False)

    id_seed_dir = p.id + "/" + "seed-{}".format(p.seed) + "/"
    ckpt_dir = "models/" + id_seed_dir

    with tf.Session(config=config) as sess:

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

        sess.run(tf.global_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists,
            # restore the parameters
            # and set epoch_n and i_iter
            # print("Loaded model: ", ckpt.model_checkpoint_path)
            model.g['saver'].restore(sess, ckpt.model_checkpoint_path)
            results['checkpoint'] = extract_model_name_from_path(ckpt.model_checkpoint_path)
            # results['checkpoint'] = ckpt.model_checkpoint_path

            eval_par = {'batch_size': p.batch_size}
            x = model.g['images']
            y = model.g['labels']
            model_preds = model.g['softmax'] # softmaxed

            X_test = dataset.test.images
            Y_test = dataset.test.labels
            acc = model_eval(
                sess, x=x, y=y,
                predictions=model_preds,
                X_test=X_test, Y_test=Y_test,
                # X_test= dataset.train.labeled_ds.images,
                # Y_test=dataset.train.labeled_ds.labels,
                feed={model.g['train_flag']: False}, args=eval_par)

            aer = 100 * (1-acc)
            results['normal_aer'] = aer
            # print('Test AER on normal examples: {:0.4f} %'.format(aer))

            if p.ord == 'inf':
                import numpy as np
                ord = np.inf
            elif p.ord == '1':
                ord = 1
            else:
                ord = 2

            fgsm_params = {'eps': 0.3, 'ord': ord}
            fgsm = FastGradientMethod(model, sess=sess)

            adv_x = fgsm.generate(x, **fgsm_params)
            adv_x = tf.stop_gradient(adv_x)
            preds_adv = model.get_probs(adv_x)

            # Evaluate the accuracy of the MNIST model on adversarial examples
            acc = model_eval(sess, x, y, preds_adv, X_test, Y_test,
                             feed={model.g['train_flag']: False},
                             args=eval_par)
            aer = 100 * (1 - acc)
            results['adv_aer'] = aer
            # print('Test AER on adversarial examples: {:0.4f} %'.format(aer))
            print(results['checkpoint'], results['normal_aer'], results[
                'adv_aer'], sep=',')

    return results



class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

def test_adversarial():

    results = {}
    p = process_params(get_cli_params())

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.which_gpu)
    # p.epsilon = parse_argstring('1.0-0.1-0.001-0.001-0.001-0.001-0.001', float)
    p.rc_weights = parse_argstring('1000-10-0.1-0.1-0.1-0.1-0.1', float)
    # p.num_labeled = 50
    p.batch_size = 50 if p.num_labeled is 50 else 100
    p.input_size = 784
    p.encoder_layers = [p.input_size, 1000, 500, 250, 250, 250, 10]
    p.lrelu_a = 0.1
    seeds = [8340, 8794, 2773, 967, 2368]
    # models = ['n', 'nlw', 'ladder', 'c', 'clw', 'vat']
    model = 'vat'
    nl = [50, 100, 1000]
    for n in nl:
        p.model = model
        p.num_labeled = n
        results[model] = {}
        for seed in seeds:
            p.seed = seed
            p.id = 'full_{}_labeled-{}'.format(model, p.num_labeled)
            # p.id = '{}{}'.format(model,p.num_labeled)
            results[model][seed] = test_aer_on_normal_and_adv(p)

    save_obj(results, 'labeled-{}'.format(p.num_labeled))
    return results

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def unpack(res):
    from statistics import mean, stdev
    for m, md in res.items():
        adv = []
        test = []
        for s, sd in md.items():
            adv.append(sd['adv_aer'])
            test.append(sd['normal_aer'])
        print(m, *['{:4.4f}'.format(x) for x in [mean(test), stdev(test),
                                           mean(adv), stdev(adv)]])




if __name__ == '__main__':
    # p = process_cli_params(get_cli_params())
    # main(p)
    results = test_adversarial()

