import os
import tensorflow as tf

import config.system as sys_config
from vagan.network_zoo.nets2D import critics, mask_generators

# Experiment name
experiment_name = 'synth_vagan_rerun'

# Model settings
critic_net = critics.C3D_fcn_16_2D
generator_net = mask_generators.unet_16_2D_bn

# Data settings
data_identifier = 'synthetic'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/synthetic')
image_size = [112, 112]
effect_size = 100
num_samples = 10000
moving_effect = True
rescale_to_one = True

# Optimizer Settings
optimizer_handle = tf.train.AdamOptimizer
beta1 = 0.0
beta2 = 0.9

# Training settings
batch_size = 32
n_accum_grads = 1
learning_rate = 1e-4  # Used 1e-3 for experiments in paper, but 1e-4 works a bit better
divide_lr_frequency = None
critic_iter = 5
critic_iter_long = 100
critic_retune_frequency = 100
critic_initial_train_duration = 25

# Cost function settings
l1_map_weight = 100.0
use_tanh = True  # Using the tanh activation function at the output of the generator will scale the outputs in the
                 # range of [-1, 1]. This can make training a bit more stable (obviously the input data must be scaled
                 # accordingly.

# Improved training settings
improved_training = True
scale=10.0

# Normal WGAN training settings (only used if improved_training=False)
clip_min = -0.01
clip_max = 0.01

# Rarely changed settings
max_iterations = 100000
save_frequency = 10
validation_frequency = 10
num_val_batches = 20
update_tensorboard_frequency = 2
