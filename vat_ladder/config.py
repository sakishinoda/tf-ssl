from argparse import Namespace

p = Namespace()

p.id                =   "C"
p.logdir            =   "train/cifar10/"
p.ckptdir           =   "train/cifar10/"
p.write_to          =   "description"
p.do_not_save       =   None
p.verbose           =   True

p.dataset	        =	"cifar10"
p.input_size        =   784

p.test_frequency_in_epochs	=	1
p.validation	            =	0
p.tb                        =   False

p.which_gpu     =   0
p.seed          =   8340
p.end_epoch     =   20
p.num_labeled   =   4000
p.batch_size    =   100
p.ul_batch_size =   156

p.initial_learning_rate =   0.002
p.decay_start           =   0.5
p.lr_decay_frequency    =   5
p.beta1                 =   0.9
p.beta1_during_decay    =   0.9

p.encoder_layers	=	"1000-500-250-250-250-10"
p.corrupt_sd	    =	0.3
p.rc_weights        =   "0-0-0-0-0-0-0-0-0-0-0-0-0-4.0"
p.static_bn	        =	0.99
p.lrelu_a	        =	0.1
p.top_bn            =   False

p.epsilon           =   "8.0"
p.num_power_iters	=	3
p.xi	            =	1e-6
p.vadv_sd	        =	0.5

p.model                 =   "n"
p.decoder               =   "gamma"  # gamma, full, or None
p.measure_smoothness    =   False
p.measure_vat           =   False

p.cnn               =   True
p.cnn_layer_types   = \
    "c-c-c-max-c-c-c-max-c-c-c-avg-fc"
p.cnn_fan           = \
    "3-96-96-96-96-192-192-192-192-192-192-192-192-10"
p.cnn_ksizes        = \
    "3-3-3-3-3-3-3-3-3-1-1-0-0"
p.cnn_strides       = \
    "1-1-1-2-1-1-1-2-1-1-1-0-0"
p.cnn_dims          = \
    "32-32-32-32-16-16-16-16-8-8-8-8-1"

