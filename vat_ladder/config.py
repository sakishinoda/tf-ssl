from argparse import Namespace

p = Namespace()

p.id                =   "gamma_n"
p.logdir            =   "train/conv_large/"
p.ckptdir           =   "train/conv_large/"
p.write_to          =   "description"
p.do_not_save       =   False
p.verbose           =   True

p.dataset	        =	"cifar10"
p.input_size        =   784

p.test_frequency_in_epochs	=	1
p.validation	            =	0
p.tb                        =   False

p.which_gpu     =   0
p.seed          =   8340
p.end_epoch     =   70
p.num_labeled   =   4000
p.batch_size    =   100
p.ul_batch_size =   156

p.initial_learning_rate =   0.002
p.decay_start           =   0.86
p.lr_decay_frequency    =   1
p.beta1                 =   0.9
p.beta1_during_decay    =   0.9

p.encoder_layers	=	"1000-500-250-250-250-10"
p.input_noise_sd    =   0.3
p.corrupt_sd	    =	0.3
p.rc_weights        =   "0-0-0-0-0-0-0-0-0-0-0-0-4.0"
p.static_bn	        =	0.99
p.lrelu_a	        =	0.1
p.top_bn            =   True

p.epsilon           =   "8.0-0-0-0-0-0-0-0-0-0-0-0-0-0"
p.num_power_iters	=	3
p.xi	            =	1e-6
p.vadv_sd	        =	0.5

p.model                 =   "n"
p.decoder               =   "gamma"  # gamma, full, or None
p.measure_smoothness    =   False
p.measure_vat           =   False

p.cnn               =   True
p.cnn_layer_types   = \
    "cv-cf-cf-max-cv-cf-cv-max-cv-cv-cv-avg"
p.cnn_fan           = \
    "3-96-96-96-96-192-192-192-192-192-192-10-10"
p.cnn_ksizes        = \
    "3-3-3-3-3-3-3-3-3-1-1-0"
p.cnn_strides       = \
    "1-1-1-2-1-1-1-2-1-1-1-0"
p.cnn_dims          = \
    "32-32-32-32-16-16-16-16-8-8-8-1"

