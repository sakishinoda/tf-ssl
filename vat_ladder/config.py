from argparse import Namespace

p = Namespace()

p.id                =   "gamma"
p.logdir            =   "train/svhn/"
p.ckptdir           =   "train/svhn/"
p.write_to          =   "description"
p.do_not_save       =   False
p.verbose           =   True

p.dataset	        =	"svhn"

p.test_frequency_in_epochs	=	1
p.validation	            =	0
p.tb                        =   False

p.which_gpu     =   None
p.seed          =   8340
p.end_epoch     =   70
p.num_labeled   =   1000
p.batch_size    =   100
p.ul_batch_size =   100

p.initial_learning_rate =   0.002
p.decay_start           =   0.86
p.lr_decay_frequency    =   1
p.beta1                 =   0.9
p.beta1_during_decay    =   0.9

p.encoder_layers	=	"1000-500-250-250-250-10"
p.input_noise_sd    =   0.3
p.corrupt_sd	    =	0.3
p.rc_weights        =   {0: 0.0,
                         1: 0.0,
                         2: 0.0,
                         3: 0.0,
                         4: 0.0,
                         5: 0.0,
                         6: 0.0,
                         7: 0.0,
                         8: 0.0,
                         9: 0.0,
                         10: 0.0,
                         11: 0.0,
                         12: 0.0,
                         13: 4.0}
p.static_bn	        =	0.99
p.lrelu_a	        =	0.1
p.top_bn            =   True

p.epsilon           =   "8.0-0-0-0-0-0-0-0-0-0-0-0-0-0"
p.num_power_iters	=	3
p.xi	            =	1e-6
p.vadv_sd	        =	0.5

p.model                 =   "ladder"  # c, clw, n, nlw, ladder, vat
p.decoder               =   "gamma"  # gamma, full, or None
p.measure_smoothness    =   False
p.measure_vat           =   False

p.cnn               =   True
p.cnn_layer_types   =   ['c', 'c', 'c', 'max', 'c', 'c', 'c', 'max', 'cv', 'c', 'c', 'avg', 'fc']
p.cnn_fan           =   [64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 10]
p.cnn_ksizes        =   [3, 3, 3, 2, 3, 3, 3, 2, 3, 1, 1, 6, 1]
p.cnn_strides       =   [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1]
p.cnn_dims          =   [32, 32, 32, 32, 16, 16, 16, 16, 8, 6, 6, 6, 1, ]


# Checks
p_dict = vars(p)
if p.cnn:
    num_layers = len(p.cnn_layer_types)
    for argname in ['cnn_layer_types', 'cnn_fan', 'cnn_ksizes', 'cnn_strides',
                'cnn_dims']:
        arg = p_dict[argname]
        assert len(arg) == num_layers, "Length of {} is {} vs expected {}".format(
            argname, len(arg), num_layers)
        assert isinstance(arg, list), "{} is a {}".format(argname, type(arg))

else:
    assert isinstance(p.encoder_layers, list), "Encoder layer spec is not a " \
                                               "list"
    num_layers = len(p.encoder_layers)

num_layers += 1
if p.model != 'ladder':
    assert isinstance(p.epsilon, dict)
    assert len(p.epsilon.keys()) == num_layers, \
        "{} epsilons vs expected {}".format(len(p.epsilon.keys()), num_layers)

elif p.model != 'vat':
    assert isinstance(p.rc_weights, dict)
    assert len(p.rc_weights.keys()) == num_layers, \
        "{} weights vs expected {}".format(len(p.rc_weights.keys()),
                                            num_layers)


# CIFAR10
# p.cnn_layer_types   = \
#     "cv-cf-cf-max-cv-cf-cv-max-cv-cv-cv-avg"
# p.cnn_fan           = \
#     "3-96-96-96-96-192-192-192-192-192-192-10-10"
# p.cnn_ksizes        = \
#     "3-3-3-3-3-3-3-3-3-1-1-0"
# p.cnn_strides       = \
#     "1-1-1-2-1-1-1-2-1-1-1-0"
# p.cnn_dims          = \
#     "32-30-32-34-16-14-16-14-6-4-4-4"
    # "32-32-32-32-16-16-16-16-8-8-8-1"


