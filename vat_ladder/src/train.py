
def evaluate_metric(dataset, sess, op, graph, params):
    metric = 0
    num_eval_iters = dataset.num_examples // params.eval_batch_size
    for _ in range(num_eval_iters):
        images, labels = dataset.next_batch(params.eval_batch_size)
        init_feed = {graph['images']: images,
                     graph['labels']: labels,
                     graph['train_flag']: False}
        metric += sess.run(op, init_feed)
    metric /= num_eval_iters
    return metric

def evaluate_metric_list(dataset, sess, ops, graph, params):
    metrics = [0.0 for _ in ops]
    num_eval_iters = dataset.num_examples // params.eval_batch_size
    for _ in range(num_eval_iters):
        images, labels = dataset.next_batch(params.eval_batch_size)
        init_feed = {graph['images']: images,
                     graph['labels']: labels,
                     graph['train_flag']: False}
        op_eval = sess.run(ops, init_feed)

        for i, op in enumerate(op_eval):
            metrics[i] += op

    metrics = [metric/num_eval_iters for metric in metrics]
    return metrics

def update_decays(sess, epoch_n, iter, graph, params):
    # ---------------------------------------------
    # Update batch norm decay constant
    if params.static_bn is False:
        sess.run(graph['ladder'].bn_decay.assign(1.0 - (1.0 / (epoch_n + 1))))

    # ---------------------------------------------
    # Update learning rate every epoch
    if ((epoch_n + 1) >= params.decay_start_epoch) and ((iter + 1) % (
                params.iter_per_epoch * params.lr_decay_frequency) == 0):
        # epoch_n + 1 because learning rate is set for next epoch
        ratio = 1.0 * (params.end_epoch - (epoch_n + 1))
        ratio = max(0., ratio / (params.end_epoch - params.decay_start_epoch))
        sess.run(graph['lr'].assign(params.initial_learning_rate *
                                    ratio))
