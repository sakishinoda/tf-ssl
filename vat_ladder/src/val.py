"""Virtual Adversarial Ladder"""
import tensorflow as tf
from src.utils import count_trainable_params, preprocess, get_batch_ops
from src.vat import Adversary
from src.ladder import Ladder

def build_top_graph(params):

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
    adv = Adversary(ladder.bn, params, start_layer=0)
    join, split_lu, labeled, unlabeled = get_batch_ops(params.batch_size)

    # AT on labeled only
    # at_cost = adversarial_loss(x=labeled(inputs),
    #                            y=outputs,
    #                            loss=ladder.cost,
    #                            is_training=train_flag,
    #                            start_layer=0
    #                            ) * params.at_weight
    # at_cost = tf.zeros(shape=tf.shape(labeled(inputs)))

    # VAT on unlabeled only
    vat_cost = adv.virtual_adversarial_loss(
        x=unlabeled(inputs),
        logit=unlabeled(ladder.corr.logits),
        is_training=train_flag) * params.vat_weight

    # -----------------------------
    # Loss, accuracy and training steps
    loss = ladder.cost + ladder.u_cost + vat_cost
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(ladder.predict, tf.argmax(outputs, 1)),
            "float")) * tf.constant(100.0)

    learning_rate = tf.Variable(params.initial_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate,
                                        beta1=params.beta1).minimize(loss)

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

    # Metrics
    m = dict()
    m['loss'] = loss
    m['cost'] = ladder.cost
    m['uc'] = ladder.u_cost
    m['vc'] = vat_cost
    m['acc'] = accuracy

    trainable_params = count_trainable_params()

    return g, m, trainable_params


def build_lw_graph(params):

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
    join, split_lu, labeled, unlabeled = get_batch_ops(params.batch_size)

    vat_costs = []
    for l in range(ladder.num_layers):
        adv = Adversary(ladder.bn, params, start_layer=l)
        # VAT on unlabeled only
        vat_costs.append(
            adv.virtual_adversarial_loss(
                x=ladder.corr.unlabeled.z[l],
                logit=unlabeled(ladder.corr.logits),
                is_training=train_flag) *
            params.rc_weights[l]
        )
    vat_cost = tf.add_n(vat_costs) * (params.vat_weight/ladder.num_layers)

    # -----------------------------
    # Loss, accuracy and training steps
    loss = ladder.cost + ladder.u_cost + vat_cost
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(ladder.predict, tf.argmax(outputs, 1)),
            "float")) * tf.constant(100.0)

    learning_rate = tf.Variable(params.initial_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate,
                                        beta1=params.beta1).minimize(loss)

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

    # Metrics
    m = dict()
    m['loss'] = loss
    m['cost'] = ladder.cost
    m['uc'] = ladder.u_cost
    m['vc'] = vat_cost
    m['acc'] = accuracy

    trainable_params = count_trainable_params()

    return g, m, trainable_params
