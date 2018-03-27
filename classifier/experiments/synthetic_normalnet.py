from classifier.network_zoo import nets2D
import tensorflow as tf
import os
import config.system as sys_config

experiment_name = 'synth_normalnet'

# Model settings
classifier_net = nets2D.normalnet2D

# Data settings
data_identifier = 'synthetic'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/synthetic')
image_size = [112, 112]
effect_size = 100
num_samples = 10000
moving_effect = True
rescale_to_one = True
nlabels = 2

# Cost function
weight_decay = 0.0
use_sigmoid = False

# Training settings
batch_size = 30
n_accum_grads = 1
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
schedule_lr = False
divide_lr_frequency = None
warmup_training = False
momentum = None

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_iterations = 1000000
train_eval_frequency = 500
val_eval_frequency = 100
