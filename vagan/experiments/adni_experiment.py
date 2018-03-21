import os
import tensorflow as tf

from config import system as sys_config
from vagan.network_zoo.nets3D import mask_generators, critics

# Experiment name
experiment_name = 'debug_rerun'

# Model settings
critic_net = critics.C3D_fcn_16
generator_net = mask_generators.unet_16_bn

# Data settings
data_identifier = 'adni'
image_size = (128, 160, 112)
target_resolution =  (1.3, 1.3, 1.3)
offset = None
label_list = (1,2)  # 0 - normal, 1 - mci, 2 - alzheimer's
label_name = 'diagnosis'
nlabels = len(label_list)
data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ADNI_Christian/ADNI_allfixed_allPP_robex'
preproc_folder = os.path.join(sys_config.project_root,'data/preproc_data/allfixed_noskull')
rescale_to_one = True
image_z_slice = 56  # for displaying images during training

# Optimizer Settings
optimizer_handle = tf.train.AdamOptimizer
beta1 = 0.0
beta2 = 0.9

# Training settings
batch_size = 2
n_accum_grads = 6
learning_rate = 1e-3
divide_lr_frequency = None
critic_iter = 5
critic_iter_long = 100
critic_retune_frequency = 100
critic_initial_train_duration = 25

# Cost function settings
l1_map_weight = 100.0
use_tanh = True

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
