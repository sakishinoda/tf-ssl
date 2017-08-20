"""Virtual Adversarial Ladder"""
import tensorflow as tf
from src.utils import count_trainable_params, preprocess, get_batch_ops
from src.vat import Adversary
from src.ladder import Ladder, Encoder, Decoder

class LadderWithVAN(Ladder):
    def get_corrupted_encoder(self, inputs, bn, train_flag, params,
                              start_layer=0, update_batch_stats=False,
                              scope='enc', reuse=True):
        return VANEncoder(
            inputs, bn, train_flag, params, self.clean.logits,
            this_encoder_noise=params.corrupt_sd,
            start_layer=start_layer, update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse)


class VANEncoder(Encoder):
    def __init__(
            self, inputs, bn, is_training, params, clean_logits,
            this_encoder_noise=0.0, start_layer=0, update_batch_stats=True,
            scope='enc', reuse=None):

        self.params = params
        self.clean_logits = clean_logits

        super(VANEncoder, self).__init__(
            inputs, bn, is_training, params,
            this_encoder_noise=this_encoder_noise,
            start_layer=start_layer,
            update_batch_stats=update_batch_stats,
            scope=scope, reuse=reuse
        )

    def get_vadv_noise(self, inputs, l_out):
        join, split_lu, labeled, unlabeled = get_batch_ops(self.batch_size)

        adv = Adversary(
            bn=self.bn,
            params=self.params,
            layer_eps=self.params.epsilon[l_out-1],
            start_layer=l_out-1
        )

        x = unlabeled(inputs)
        logit = unlabeled(self.clean_logits)

        ul_noise = adv.generate_virtual_adversarial_perturbation(
            x=x, logit=logit, is_training=self.is_training)

        return join(tf.zeros(tf.shape(labeled(inputs))), ul_noise)

    def print_progress(self, l_out):
        el = self.encoder_layers
        print("Layer {}: {} -> {}, epsilon {}".format(l_out, el[l_out - 1], el[l_out],
              self.params.epsilon.get(l_out - 1)))

    def generate_noise(self, inputs, l_out):
        print("Generating noise for layer", l_out)
        if self.noise_sd > 0.0:
            noise = tf.random_normal(tf.shape(inputs)) * self.noise_sd

            if self.params.model == "n" and l_out==0:
                noise += self.get_vadv_noise(inputs, l_out)

            elif self.params.model == "nlw":
                noise += self.get_vadv_noise(inputs, l_out)

        else:
            noise = tf.zeros(tf.shape(inputs))

        return inputs + noise



def get_vat_cost(ladder, train_flag, params):
    unlabeled = lambda x: x[params.batch_size:] if x is not None else x

    def get_layer_vat_cost(l):

        adv = Adversary(bn=ladder.bn,
                        params=params,
                        layer_eps=params.epsilon[l],
                        start_layer=l)

        # VAT on unlabeled only
        return (
            adv.virtual_adversarial_loss(
                x=ladder.corr.unlabeled.z[l],
                logit=unlabeled(ladder.corr.logits),  # should this be clean?
                is_training=train_flag)
        )

    if params.model == "clw":
        vat_costs = []
        for l in range(ladder.num_layers):
            vat_costs.append(get_layer_vat_cost(l))
        vat_cost = tf.add_n(vat_costs)

    elif params.model == "c":
        vat_cost = get_layer_vat_cost(0)

    else:
        vat_cost = 0.0

    return vat_cost


def build_graph(params):
    model = params.model
    # -----------------------------
    # Placeholder setup
    inputs_placeholder = tf.placeholder(
        tf.float32, shape=(None, params.encoder_layers[0]))
    inputs = preprocess(inputs_placeholder, params)
    outputs = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)


    if model == "c" or model == "clw":
        ladder = Ladder(inputs, outputs, train_flag, params)
        vat_cost = get_vat_cost(ladder, train_flag, params)

    elif model == "n" or model == "nlw":
        ladder = LadderWithVAN(inputs, outputs, train_flag, params)
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
