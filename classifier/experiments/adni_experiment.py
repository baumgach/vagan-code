
from classifier.network_zoo import nets3D
import tensorflow as tf
import config.system as sys_config
import os

experiment_name = 'adni_experiment'

classifier_net = nets3D.FCN_32_bn
multi_task_model = True

# Data settings
data_identifier = 'adni'
image_size = (128, 160, 112)  #(64, 80, 56)
target_resolution =  (1.3, 1.3, 1.3)
offset = None
label_list = (1,2)  # 0 - normal, 1 - mci, 2 - alzheimer's
label_name = 'diagnosis'
nlabels = len(label_list)
data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ADNI_Christian/ADNI_allfixed_allPP_robex'
preproc_folder = os.path.join(sys_config.project_root,'data/preproc_data/allfixed_noskull')
rescale_to_one = False

# Cost function
weight_decay = 0.0

# Training settings
batch_size = 3
n_accum_grads = 10
learning_rate = 1e-4
divide_lr_frequency = None
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
schedule_lr = False
warmup_training = False
momentum = None

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_iterations = 1000000
train_eval_frequency = 500
val_eval_frequency = 100
