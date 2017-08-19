"""Virtual Adversarial Ladder"""
import tensorflow as tf
from src.utils import count_trainable_params, preprocess, get_batch_ops
from src.vat import Adversary
from src.ladder import Ladder

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

    # -----------------------------
    # Ladder
    ladder = Ladder(inputs, outputs, train_flag, params)
    # -----------------------------
    # Add top-level VAT/AT cost
    if model == 'lw':
        vat_cost = get_lw_vat_cost(ladder, train_flag, params)
    elif model == 'top':
        vat_cost = get_top_vat_cost(ladder, train_flag, params)
    else:
        print('Help, model not defined!')
        vat_cost = tf.zeros(tf.shape(ladder.cost))

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
