"""Virtual Adversarial Ladder"""
import tensorflow as tf
from src.utils import count_trainable_params, preprocess, get_batch_ops
from src.vat import Adversary, get_normalized_vector, kl_divergence_with_logit
from src.ladder import Ladder, Encoder, VATModel, LadderWithVAN


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
        model = Ladder(inputs, outputs, train_flag, params)
        vat_cost = get_vat_cost(model, train_flag, params)
        loss = model.cost + model.u_cost + vat_cost

    elif model == "n" or model == "nlw":
        model = LadderWithVAN(inputs, outputs, train_flag, params)
        vat_cost = tf.zeros([])
        loss = model.cost + model.u_cost

    elif model == "vat":
        model = VATModel(inputs, outputs, train_flag, params)
        vat_cost = model.u_cost
        loss = model.cost + model.u_cost

    else:
        model = Ladder(inputs, outputs, train_flag, params)
        vat_cost = tf.zeros([])
        loss = model.cost + model.u_cost

    # -----------------------------
    # Loss, accuracy and training steps


    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(model.predict, tf.argmax(outputs, 1)),
            "float")) * tf.constant(100.0)

    learning_rate = tf.Variable(params.initial_learning_rate,
        name='lr', trainable=False)
    # beta1 = tf.Variable(params.beta1, name='beta1', trainable=False)
    beta1 = 0.9

    train_step = tf.train.AdamOptimizer(learning_rate,
                                        beta1=beta1).minimize(loss)

    # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*model.bn.bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5,
                            max_to_keep=5)

    # Graph
    g = dict()
    g['images'] = inputs_placeholder
    g['labels'] = outputs
    g['train_flag'] = train_flag
    g['ladder'] = model
    g['saver'] = saver
    g['train_step'] = train_step
    g['lr'] = learning_rate
    g['beta1'] = beta1

    # Metrics
    m = dict()
    m['loss'] = loss
    m['cost'] = model.cost
    m['uc'] = model.u_cost * 0.0
    m['acc'] = accuracy
    m['vc'] = vat_cost

    trainable_params = count_trainable_params()

    return g, m, trainable_params



def get_spectral_radius(x, logit, forward, num_power_iters=1, xi=1e-6):

    prev_d = tf.random_normal(shape=tf.shape(x))
    for k in range(num_power_iters):
        d = xi * get_normalized_vector(prev_d)
        logit_p = logit
        logit_m = forward(x + d)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        prev_d = tf.stop_gradient(grad)

    prev_d, d = get_normalized_vector(prev_d), get_normalized_vector(d)
    dot = lambda a, b: tf.reduce_mean(tf.multiply(a, b), axis=1)
    return dot(d, prev_d)/dot(prev_d, prev_d)



def measure_smoothness(g, params):
    # Measure smoothness using clean logits
    print("=== Measuring smoothness ===")
    inputs = g['images']
    logits = g['ladder'].clean.logits
    forward = lambda x: Encoder(
        inputs=x,
        bn=g['ladder'].bn,
        is_training=g['train_flag'],
        params=params,
        this_encoder_noise=0.0,
        start_layer=0,
        update_batch_stats=False,
        scope='enc',
        reuse=True
    ).logits

    return get_spectral_radius(
        x=inputs, logit=logits, forward=forward, num_power_iters=5)