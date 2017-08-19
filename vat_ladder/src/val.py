"""Virtual Adversarial Ladder"""
import tensorflow as tf
from src.utils import count_trainable_params, preprocess, get_batch_ops
from src.vat import Adversary
from src.ladder import Ladder, Encoder

class VANLWEncoder(Encoder):
    def __init__(
            self, inputs, encoder_layers, bn, is_training,
            noise_sd=0.0, start_layer=0, batch_size=100,
            update_batch_stats=True, scope='enc', reuse=None,
            epsilons=None, xi=1e-6, num_power_iters=1):
        super(VANEncoder, self).__init__(
            inputs, encoder_layers, bn, is_training,
            noise_sd=noise_sd, start_layer=start_layer, batch_size=batch_size,
            update_batch_stats=update_batch_stats, scope=scope, reuse=reuse
        )
        self.bn = bn
        self.encoder_layers = encoder_layers
        self.batch_size = batch_size
        self.eps = epsilons if epsilons is not None else \
            ((None, ) * (self.num_layers + 1))
        self.is_training = is_training
        self.xi = xi
        self.num_power_iters = num_power_iters


    def generate_noise(self, inputs, l):
        adv = Adversary(
            bn=self.bn,
            encoder_layers=self.encoder_layers,
            batch_size=self.batch_size,
            epsilon=self.eps[l],
            xi=self.xi,
            num_power_iters=self.num_power_iters,
            start_layer=l,
            encoder_class=VANEncoder
        )
        noise = tf.random_normal(tf.shape(inputs)) * self.noise_sd
        noise += adv.generate_virtual_adversarial_perturbation(
            inputs, self.is_training)

        return inputs + noise


class VANEncoder(Encoder):
    def __init__(
            self, inputs, encoder_layers, bn, is_training,
            noise_sd=0.0, start_layer=0, batch_size=100,
            update_batch_stats=True, scope='enc', reuse=None,
            epsilons=None, xi=1e-6, num_power_iters=1):
        super(VANEncoder, self).__init__(
            inputs, encoder_layers, bn, is_training,
            noise_sd=noise_sd, start_layer=start_layer, batch_size=batch_size,
            update_batch_stats=update_batch_stats, scope=scope, reuse=reuse
        )
        self.bn = bn
        self.encoder_layers = encoder_layers
        self.batch_size = batch_size
        self.eps = epsilons if epsilons is not None else \
            ((None, ) * (self.num_layers + 1))
        self.is_training = is_training
        self.xi = xi
        self.num_power_iters = num_power_iters


    def generate_noise(self, inputs, l):
        noise = tf.random_normal(tf.shape(inputs)) * self.noise_sd

        if l == 0:
            adv = Adversary(
                bn=self.bn,
                encoder_layers=self.encoder_layers,
                batch_size=self.batch_size,
                epsilon=self.eps[l],
                xi=self.xi,
                num_power_iters=self.num_power_iters,
                start_layer=l,
                encoder_class=VANEncoder
            )
            noise += adv.generate_virtual_adversarial_perturbation(
                inputs, self.is_training)

        return inputs + noise


def get_lw_vat_cost(ladder, train_flag, params):
    unlabeled = lambda x: x[params.batch_size:] if x is not None else x

    vat_costs = []
    for l in range(ladder.num_layers):
        adv = Adversary(bn=ladder.bn,
                        encoder_layers=params.encoder_layers,
                        batch_size=params.batch_size,
                        epsilon=params.lw_eps[l],
                        xi=params.xi,
                        num_power_iters=params.num_power_iterations,
                        start_layer=l)

        # VAT on unlabeled only
        vat_costs.append(
            adv.virtual_adversarial_loss(
                x=ladder.corr.unlabeled.z[l],
                logit=unlabeled(ladder.corr.logits),
                is_training=train_flag)
        )
    vat_cost = tf.add_n(vat_costs)
    return vat_cost


def get_top_vat_cost(ladder, train_flag, params):
    unlabeled = lambda x: x[params.batch_size:] if x is not None else x
    adv = Adversary(bn=ladder.bn,
                    encoder_layers=params.encoder_layers,
                    batch_size=params.batch_size,
                    epsilon=params.epsilon,
                    xi=params.xi,
                    num_power_iters=params.num_power_iterations,
                    start_layer=0)

    # VAT on unlabeled only
    vat_cost = \
        adv.virtual_adversarial_loss(
            x=ladder.corr.unlabeled.z[0],
            logit=unlabeled(ladder.corr.logits),
            is_training=train_flag)
    return vat_cost


def build_graph(params, model='top'):

    # -----------------------------
    # Placeholder setup
    inputs_placeholder = tf.placeholder(
        tf.float32, shape=(None, params.encoder_layers[0]))
    inputs = preprocess(inputs_placeholder, params)
    outputs = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)


    if model == 'lw':
        # Layer-wise VAT costs
        ladder = Ladder(inputs, outputs, train_flag, params)
        vat_cost = get_lw_vat_cost(ladder, train_flag, params)

    elif model == 'top':
        # Add top-level VAT/AT cost
        ladder = Ladder(inputs, outputs, train_flag, params)
        vat_cost = get_top_vat_cost(ladder, train_flag, params)

    elif model == 'noise':
        # Add Virtual Adversarial Noise at each layer
        ladder = Ladder(inputs, outputs, train_flag, params, encoder=VANEncoder)
        vat_cost = 0.0

    else:

        ladder = Ladder(inputs, outputs, train_flag, params)
        vat_cost = 0.0


    # -----------------------------
    # Loss, accuracy and training steps
    loss = ladder.cost + ladder.u_cost + vat_cost
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(ladder.predict, tf.argmax(outputs, 1)),
            "float")) * tf.constant(100.0)

    learning_rate = tf.Variable(params.initial_learning_rate, trainable=False)
    beta1 = tf.Variable(params.beta1, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate,
                                        beta1=beta1).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*ladder.bn.bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5,
                            max_to_keep=5)

    # Graph
    g = dict()
    g['images'] = inputs_placeholder
    g['labels'] = outputs
    g['train_flag'] = train_flag
    g['ladder'] = ladder
    g['saver'] = saver
    g['train_step'] = train_step
    g['lr'] = learning_rate
    g['beta1'] = beta1

    # Metrics
    m = dict()
    m['loss'] = loss
    m['cost'] = ladder.cost
    m['uc'] = ladder.u_cost
    m['vc'] = vat_cost
    m['acc'] = accuracy

    trainable_params = count_trainable_params()

    return g, m, trainable_params
