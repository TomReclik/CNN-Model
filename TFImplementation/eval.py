from __future__ import print_function, division
import tensorflow as tf
import utils

##
## enable logging
##

tf.logging.set_verbosity(tf.logging.INFO)

##
## command line args
##

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/tensorflow/train', 'Directory where to write event logs and checkpoints ')
tf.app.flags.DEFINE_integer('n_epoch', 1000, 'Number of training steps to run.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Size of mini batches')
tf.app.flags.DEFINE_float('initial_learn_rate', 0.01, 'Initial learn rate')
tf.app.flags.DEFINE_integer('decay_steps', 100, 'Update learn rate after n steps')
tf.app.flags.DEFINE_float('decay_rate', 0.5, 'Decay rate for learning rate adaption')

(x_train,y_train,x_val,y_val,x_test,y_test) = utils.loadCIFAR10(size=5000,labels=[0,1,2,3,4],freq=0,val_split=0.2)
