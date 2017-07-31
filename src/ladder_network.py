import tensorflow as tf
from collections import OrderedDict
from tensorflow.examples.tutorials.mnist import input_data
from time import time
from src import feed, utils
import numpy as np
import sys, os
from src.utils import fclayer, lrelu


class NoisyBNLayer(object):

    def __init__(self, scope_name, size, noise_sd=None, decay=0.99, var_ep=1e-5, reuse=None):
        self.scope = scope_name
        self.size = size
        self.noise_sd = noise_sd
        self.decay = decay
        self.var_ep = var_ep
        with tf.variable_scope(scope_name, reuse=reuse):
            self.scale = tf.get_variable('NormScale', initializer=tf.ones([size]))
            self.beta = tf.get_variable('NormOffset', initializer=tf.zeros([size]))
            self.pop_mean = tf.get_variable('PopulationMean', initializer=tf.zeros([size]), trainable=False)
            self.pop_var = tf.get_variable('PopulationVariance', initializer=tf.fill([size], 1e-2), trainable=False)
        self.batch_mean, self.batch_var = None, None

    def normalize(self, x, training=True):
        eps = self.var_ep
        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0])
            self.batch_mean, self.batch_var = batch_mean, batch_var
            train_mean_op = tf.assign(self.pop_mean,
                                      self.pop_mean * self.decay + batch_mean * (1 - self.decay))
            train_var_op = tf.assign(self.pop_var,
                                     self.pop_var * self.decay + batch_var * (1 - self.decay))

            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.divide(tf.subtract(x, batch_mean), tf.add(tf.sqrt(batch_var), eps))

        else:
            return tf.divide(tf.subtract(x, self.pop_mean), tf.add(tf.sqrt(self.pop_var), eps))

    def normalize_from_saved_stats(self, x, training=True):
        if training:
            return tf.divide(tf.subtract(x, self.batch_mean), tf.add(tf.sqrt(self.batch_var), self.var_ep))
        else:
            return tf.divide(tf.subtract(x, self.pop_mean), tf.add(tf.sqrt(self.pop_var), self.var_ep))

    def add_noise(self, x):
        if self.noise_sd is not None:
            noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=self.noise_sd)
            return x + noise
        else:
            return x

    def apply_shift_scale(self, x, shift=True, scale=True):
        if shift:
            x = tf.add(x, self.beta)
        if scale:
            x = tf.multiply(self.scale, x)
        return x

    def normalize_and_add_noise(self, x, labeled_batch_size, training=True):
        return self.add_noise(
            tf.stack(
                [self.normalize(x[:labeled_batch_size], training), # labeled
                 self.normalize_from_saved_stats(x[labeled_batch_size:],
                                                 training)], # unlabeled
                axis=0
            )
        )



class Encoder(object):
    def __init__(self, x, y, layer_sizes, noise_sd=None, reuse=None,
                 training=True, batch_size=100):
        """
        """
        self.scope = 'enc'
        self.reuse = reuse
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.labeled_batch_size = batch_size  # testing at test time
        self.last = self.n_layers-1

        self.x = x
        self.z_pre = OrderedDict()
        self.z = OrderedDict()
        self.h = OrderedDict()
        self.y = y


        self.bn_layers = [NoisyBNLayer(scope_name=self.scope+'_bn'+str(i),
                                       size=layer_sizes[i],
                                       noise_sd=noise_sd,
                                       reuse=reuse) for i in range(self.n_layers)]


        self.predict = self.build(training)


    def build(self, training=True):
        """
        Defines all operations needed for inference.
        Supervised loss also required for training.
        Note that z[0], z_pre[0] not defined; h[0] is the original input

        """
        labeled_batch_size = self.labeled_batch_size

        bn = self.bn_layers[0]
        self.h[0] = bn.normalize_and_add_noise(self.x, labeled_batch_size,
                                               training)
        self.z[0] = self.h[0]

        for l in range(1, self.n_layers):
            # print('enc', l)
            size_out = self.layer_sizes[l]
            bn = self.bn_layers[l]
            self.z_pre[l] = fclayer(self.h[l-1], size_out, reuse=self.reuse, scope=self.scope+str(l))

            if l == self.n_layers - 1:
                self.z[l] = bn.normalize_and_add_noise(self.z_pre[l],
                                                       labeled_batch_size,
                                                       training)
                self.h[l] = tf.nn.softmax(logits=bn.apply_shift_scale(self.z[l], shift=True, scale=True))
            else:
                # Do not need to apply scaling to RELU
                self.z[l] = bn.normalize_and_add_noise(self.z_pre[l],
                                                       labeled_batch_size,
                                                       training)
                self.h[l] = tf.nn.relu(bn.apply_shift_scale(self.z[l], shift=True, scale=False))


        predict = tf.argmax(self.h[self.n_layers-1], axis=-1)

        return predict

    def supervised_loss(self, labeled_batch_size):
        labeled_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.z[self.n_layers - 1][:labeled_batch_size])
        return tf.reduce_mean(labeled_loss, axis=0)



class Combinator(object):
    def __init__(self, decoder_input, lateral_input, layer_sizes=(2, 2, 2), stddev=0.006, scope='com'):
        """"""
        self.wts_init = tf.random_normal_initializer(stddev=stddev)
        self.bias_init = tf.truncated_normal_initializer(stddev=1e-6)
        self.reuse = None
        self.scope = scope
        inputs = self.stack_input(decoder_input, lateral_input)
        self.outputs = self.build(inputs, layer_sizes)


    def build(self, inputs, layer_sizes):
        last = len(layer_sizes) - 1
        for l, size_out in enumerate(layer_sizes):
            inputs = fclayer(inputs, size_out, self.wts_init, self.bias_init, self.reuse, self.scope+str(l))
            if l < last:
                inputs = lrelu(inputs)
            else:
                inputs = tf.squeeze(inputs)
        return inputs


    def stack_input(self, decoder_input, lateral_input):
        # Augmented multiplicative term
        mult_input = tf.multiply(decoder_input, lateral_input)
        inputs = tf.stack([decoder_input, lateral_input, mult_input], axis=-1)
        return inputs


class Decoder(object):
    """Has n_layers (default 7) layers, keyed as range(n_layers),
    i.e. integers 0 to n_layers-1 inclusive."""
    def __init__(self, noisy, clean, scope='dec', combinator_layers=(2, 2, 1),
                 combinator_sd=0.025):

        self.noisy = noisy
        self.clean = clean
        self.scope = scope
        self.n_layers = self.noisy.n_layers

        self.combinator_layers = combinator_layers
        self.combinator_sd = combinator_sd


        self.rc_cost = OrderedDict()
        self.combinators = OrderedDict()
        self.activations = OrderedDict()
        self.reconstructions = OrderedDict()

        # label, with dim matching
        # u_l = tf.expand_dims(tf.cast(self.noisy.predict, tf.float32), axis=-1)
        # self.build(u_l)

    def build(self, decoder_activations, training=True):
        """
        Reconstruction cost is only associated with each layer of z,
        so does not include the input h[0], for which there is no
        corresponding z[0]

        :param decoder_activations:
        :param training:
        :return:
        """

        # Enumerates down from self.noisy.n_layers-1
        # n_layers = 7 by default, including inputs as 0
        # self.activations[self.n_layers] = decoder_activations
        for l in reversed([l for l in self.noisy.z.keys()]):
            # print('dec', l)
            # self.activations[l], self.rc_cost[l] = self.compute_rc_cost(
            #     l, self.activations[l+1], training)

            # # Alternative loop
            decoder_activations, self.rc_cost[l] = self.compute_rc_cost(
            l, decoder_activations, training)
            self.activations[l] = decoder_activations

    def compute_rc_cost(self, layer, decoder_activations, training=True):
        # print(layer, decoder_activations.get_shape())
        noisy, clean = self.noisy, self.clean

        # Use decoder weights to upsample the signal from above
        size_out = noisy.layer_sizes[layer]
        u_pre_l = fclayer(decoder_activations, size_out,
                          scope=self.scope + str(layer))

        u_l = noisy.bn_layers[layer].normalize_from_saved_stats(
            u_pre_l, training=training)

        # Unbatch-normalized activations from parallel layer in noisy encoder
        z_l = noisy.z[layer]

        # Unbatch-normalized target activations from parallel layer in clean encoder
        target_z = clean.z[layer]

        combinator = Combinator(
            decoder_input=u_l,
            lateral_input=z_l,
            layer_sizes=self.combinator_layers,
            stddev=self.combinator_sd,
            scope='com' + str(layer) + '_')

        reconstruction = combinator.outputs
        reconstruction.set_shape([None, size_out])

        self.combinators[layer] = combinator
        self.reconstructions[layer] = reconstruction

        # Mean over the width of the layer
        rc_cost = tf.reduce_mean(
            tf.square(reconstruction - target_z),
            axis=-1)

        return reconstruction, rc_cost


class GammaDecoder(Decoder):
    def build(self, decoder_activations, training=True):
        l = self.n_layers-1
        self.activations[l], self.rc_cost[l] = self.compute_rc_cost(
            l, decoder_activations, training=training)



class Ladder(object):

    def __init__(self, params):

        self.params = params
        encoder_layers = params.encoder_layers
        train_flag = params.train_flag
        gamma_flag = params.gamma_flag

        # ===========================
        # ENCODER
        # ===========================

        # PLACEHOLDERS
        # Input placeholder
        self.x = tf.placeholder(tf.float32, shape=(None, encoder_layers[0]))
        # One-hot targets
        self.y = tf.placeholder(tf.float32, shape=(None, encoder_layers[-1]))

        # CLEAN ENCODER
        self.clean = Encoder(self.x, self.y, encoder_layers,
                             noise_sd=None, reuse=False, training=train_flag,
                             batch_size=params.labeled_batch_size)

        # CORRUPTED ENCODER
        self.noisy = Encoder(self.x, self.y, encoder_layers,
                             noise_sd=params.encoder_noise_sd,
                             reuse=True, training=train_flag,
                             batch_size=params.labeled_batch_size)

        self.predict = self.noisy.predict

        # ===========================
        # DECODER
        # ===========================
        if gamma_flag:
            self.decoder = GammaDecoder(self.noisy, self.clean,
                                        combinator_layers=params.combinator_layers,
                                        combinator_sd=params.combinator_sd)
        else:
            self.decoder = Decoder(self.noisy, self.clean,
                                   combinator_layers=params.combinator_layers,
                                   combinator_sd=params.combinator_sd)

        self.decoder.build(
            decoder_activations=tf.expand_dims(tf.cast(self.predict, tf.float32), axis=-1),
            training=train_flag)

    @property
    def aer(self):
        """
        Compute training error rate on labeled examples only (since e.g.
        CIFAR-100 with Tiny Images, no labels are actually available)
        At test time, number of labeled examples is same as number of examples
        """

        return 1 - tf.reduce_mean(
            tf.cast(tf.equal(self.predict[:self.params.labeled_batch_size], tf.argmax(self.y, 1)), tf.float32))

    @property
    def unsupervised_loss(self):
        unlabeled_loss = sum([self.decoder.rc_cost[l][self.params.unlabeled_batch_size:] *
                    self.params.rc_weights[l] for l in self.decoder.rc_cost.keys()])
        return tf.reduce_mean(unlabeled_loss, axis=0)

    @property
    def supervised_loss(self):
        # Compute supervised loss on labeled only
        supervised_loss = self.noisy.supervised_loss(
            self.params.labeled_batch_size) * self.params.sc_weight
        return supervised_loss

    @property
    def loss(self):
        return self.loss

    @property
    def testing_loss(self):
        return self.supervised_loss

    @property
    def loss(self):
        return self.unsupervised_loss + self.supervised_loss

        # self.loss = self.supervised_loss + self.unsupervised_loss
        # self.mean_loss = tf.reduce_mean(self.loss)


# ===========================
# PARAMETERS
# ===========================
def get_params():

    params = utils.get_cli_params()

    write_to = open(params.write_to, 'w') if params.write_to is not None else None

    param_dict = vars(params)
    print('===== Parameter settings =====', flush=True, file=write_to)
    sorted_keys = sorted([k for k in param_dict.keys()])
    for k in sorted_keys:
        print(k, ':', param_dict[k], file=write_to, flush=True)

    params = utils.process_cli_params(params)

    return params, write_to


def main(params=None):

    if params is None:
        params, write_to = get_params()
    else:
        write_to = open(params.write_to, 'w') if params.write_to is not None else None

    # Specify GPU to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(params.which_gpu)

    mnist = input_data.read_data_sets(sys.path[0]+'/../../data/mnist/', one_hot=True)

    # Set tensorflow and numpy seeds
    tf.set_random_seed(params.seed)
    np.random.seed(params.seed)
    print('seed : ', params.seed, file=write_to, flush=True)

    ladder = Ladder(params)
    # IPython.embed()

    if params.train_flag:
        # ===========================
        # TRAINING
        # ===========================
        sf = feed.Balanced(
            np.concatenate([mnist.train.images, mnist.validation.images],
                           axis=0),
            np.concatenate([mnist.train.labels, mnist.validation.labels],
                           axis=0),
            params.num_labeled)
        # print('seeds : ', sf.seeds, file=write_to, flush=True)
        if params.use_labeled_epochs:
            iter_per_epoch = int(
                sf.labeled.num_examples / params.labeled_batch_size)
        else:
            iter_per_epoch = int(sf.unlabeled.num_examples / params.unlabeled_batch_size)
        print('iter_per_epoch', ':', iter_per_epoch, file=write_to, flush=True)
        utils.print_trainables(write_to=write_to)
        save_interval = None if params.save_epochs is None else int(params.save_epochs * iter_per_epoch)

        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        learning_rate = utils.decay_learning_rate(
                initial_learning_rate=params.initial_learning_rate,
                decay_start_epoch=params.decay_start_epoch,
                end_epoch=params.end_epoch,
                iter_per_epoch=iter_per_epoch,
                global_step=global_step)

        # Passing global_step to minimize() will increment it at each step.
        opt_op = tf.train.AdamOptimizer(learning_rate).minimize(ladder.loss, global_step=global_step)

        # Set up saver
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        save_to = 'models/' + params.id

        # Start session and initialize
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Create summaries
        train_writer = tf.summary.FileWriter('logs/' + params.id + '/train/', sess.graph)
        test_writer = tf.summary.FileWriter('logs/' + params.id + '/test/', sess.graph)
        train_loss_summary = tf.summary.scalar('training loss', ladder.loss)
        test_loss_summary = tf.summary.scalar('testing loss', ladder.testing_loss)
        err_summary = tf.summary.scalar('error', ladder.aer)
        train_merged = tf.summary.merge([train_loss_summary, err_summary])
        test_merged = tf.summary.merge([test_loss_summary, err_summary])

        x, y = mnist.validation.next_batch(params.labeled_batch_size)
        test_dict = {ladder.x: x, ladder.y: y}
        start_time = time()

        print('LEpoch', 'UEpoch', 'Time/m', 'Step', 'Loss', 'TestLoss', 'AER(%)', 'TestAER(%)', sep='\t', flush=True, file=write_to)
        # Training (using a separate step to count)
        end_step = params.end_epoch * iter_per_epoch
        for step in range(end_step):
            x, y = sf.next_batch(params.labeled_batch_size, params.unlabeled_batch_size)
            train_dict = {ladder.x: x, ladder.y: y}
            sess.run(opt_op, feed_dict=train_dict)

            # Logging during training
            if (step+1) % params.print_interval == 0:
                time_elapsed = time() - start_time
                labeled_epoch = sf.labeled.epochs_completed
                unlabeled_epoch = sf.unlabeled.epochs_completed

                # Training metrics
                train_summary, train_err, train_loss = \
                    sess.run([train_merged, ladder.aer, ladder.loss],
                             feed_dict=train_dict)
                train_writer.add_summary(train_summary, global_step=step)

                # Testing metrics
                test_summary, test_err, test_loss = \
                    sess.run([test_merged, ladder.aer, ladder.testing_loss],
                             feed_dict=test_dict)
                test_writer.add_summary(test_summary, global_step=step)

                print(
                    labeled_epoch,
                    unlabeled_epoch,
                    int(time_elapsed/60),
                    step,
                    train_loss,
                    test_loss,
                    train_err * 100,
                    test_err * 100,
                    sep='\t', flush=True, file=write_to)


            if (save_interval is not None
                and step > 0
                and step % save_interval == 0
                and params.do_not_save is False):
                saver.save(sess, save_to, global_step=step)

            global_step += 1

        # Save final model
        if params.do_not_save is False:
            saved_to = saver.save(sess, save_to)
            print('Model saved to: ', saved_to)

    else:

        # ===========================
        # TESTING
        # ===========================

        test_set = mnist.test
        iter_per_epoch = int(test_set.num_examples / params.num_labeled)

        # Set up saver
        saver = tf.train.Saver()
        save_to = 'models/' + params.id
        if params.train_step is not None:
            save_to += '-' + str(params.train_step)

        # Start session and initialize
        sess = tf.Session()
        # sess.run(tf.global_variables_initializer())

        saver.restore(sess, save_to)

        err = []
        loss = []

        for step in range(iter_per_epoch):
            x, y = test_set.next_batch(params.test_batch_size)
            feed_dict = {ladder.x: x, ladder.y: y}

            this_err, this_loss = \
                sess.run([ladder.aer, ladder.testing_loss], feed_dict)
            err.append(this_err)
            loss.append(this_loss)

            if params.verbose:
                print(step, loss[step], err[step] * 100, sep='\t', flush=True, file=write_to)

        mean_loss = sum(loss) / (len(loss)+1)
        mean_aer = sum(err) / (len(err)+1)

        print('Training step', 'Mean loss', 'Mean AER(%)', sep='\t', flush=True, file=write_to)
        print(params.train_step, mean_loss, mean_aer * 100, sep='\t', flush=True, file=write_to)

    sess.close()
    if write_to is not None:
        write_to.close()


if __name__ == '__main__':
    main()
