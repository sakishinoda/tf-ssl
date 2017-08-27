from cleverhans.model import Model as CleverHansModel
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from src.val import build_graph_from_inputs, build_graph
from src.utils import get_cli_params, process_cli_params, parse_argstring
from src.mnist import read_data_sets
import tensorflow as tf


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
            return g['ladder'].clean.logits

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
            num_layers = g['ladder'].clean.num_layers
            return g['ladder'].clean.labeled.h[num_layers]


def main(p):
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

    with tf.Session() as sess:

        # ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        # print(ckpt)
        # sess.run(tf.global_variables_initializer())
        # if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt_dir + "model.ckpt-249"


        # if checkpoint exists,
        # restore the parameters
        # and set epoch_n and i_iter
        print("Loaded model: ", ckpt_path)
        model.g['saver'].restore(sess, ckpt_path)
        results['checkpoint'] = ckpt_path

        eval_par = {'batch_size': p.batch_size}
        x = model.g['images']
        y = model.g['labels']
        num_layers = model.g['ladder'].clean.num_layers
        model_preds = model.g['ladder'].clean.labeled.h[num_layers] # softmaxed
        # model_preds = model.g['ladder'].clean.logits
        # import IPython
        # IPython.embed()
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
        print('Test AER on normal examples: {:0.4f} %'.format(aer))

        fgsm_params = {'eps': 0.3}
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
        print('Test AER on adversarial examples: {:0.4f} %'.format(aer))

    return results



class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

def test_labeled_50():
    results = {}
    p = process_cli_params(get_cli_params())
    p.epsilon = parse_argstring('1.0-0.1-0.001-0.001-0.001-0.001-0.001', float)
    p.rc_weights = parse_argstring('1000-10-0.1-0.1-0.1-0.1-0.1', float)
    p.num_labeled = 50
    p.batch_size = 50
    p.input_size = 784
    p.encoder_layers = [p.input_size, 1000, 500, 250, 250, 250, 10]
    seeds = [8340, 8794, 2773, 967, 2368]
    models = ['n', 'nlw', 'ladder', 'c', 'clw']
    for model in models:
        p.model = model
        results[model] = {}
        for seed in seeds:
            p.seed = seed
            p.id = 'full_{}_labeled-{}'.format(model, p.num_labeled)
            results[model][seed] = main(p)

    return results

if __name__ == '__main__':
    # p = process_cli_params(get_cli_params())
    # main(p)
    test_labeled_50()