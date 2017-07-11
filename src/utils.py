import tensorflow as tf

def decay_learning_rate(initial_learning_rate, decay_start_epoch, end_epoch, iter_per_epoch, global_step):
    end_step = end_epoch * iter_per_epoch
    decay_start_step = decay_start_epoch * iter_per_epoch

    decay_epochs = end_epoch - decay_start_epoch
    boundaries = [x for x in range(decay_start_step, end_step, iter_per_epoch)]
    decay_per_epoch = initial_learning_rate / decay_epochs
    values = [initial_learning_rate - x * decay_per_epoch for x in range(decay_epochs + 1)]
    assert len(values) == len(boundaries) + 1

    return tf.train.piecewise_constant(global_step, boundaries, values)


def parse_argstring(argstring, dtype=float, sep='-'):
    return list(map(dtype, argstring.split(sep)))