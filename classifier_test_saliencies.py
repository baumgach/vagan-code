# Code for displaying saliency maps for a trained classifier (but not VA-GAN)
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import config.system as sys_config
from classifier.model_classifier import classifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import matplotlib.pyplot as plt

def main(model_path, exp_config):

    # Get Data
    if exp_config.data_identifier == 'synthetic':
        from data.synthetic_data import synthetic_data as data_loader
    elif exp_config.data_identifier == 'adni':
        from data.adni_data import adni_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % exp_config.data_identifier)

    data = data_loader(exp_config)

    # Make and restore vagan model
    classifier_model = classifier(exp_config=exp_config, data=data, fixed_batch_size=1)

    # classifier_model.initialise_saliency(mode='additive_pertubation')
    # classifier_model.initialise_saliency(mode='backprop')
    # classifier_model.initialise_saliency(mode='integrated_gradients')
    classifier_model.initialise_saliency(mode='guided_backprop')
    # classifier_model.initialise_saliency(mode='CAM')  # Requires CAM net (obvs)

    classifier_model.load_weights(model_path, type='best_xent')

    for batch in data.testAD.iterate_batches(1):

        x, y = batch

        sal = classifier_model.compute_saliency(x, label=1)

        plt.figure()
        plt.imshow(np.squeeze(x))

        plt.figure()
        plt.imshow(np.squeeze(sal))
        plt.show()



if __name__ == '__main__':

    base_path = sys_config.project_root

    # Code for selecting experiment from command line
    # parser = argparse.ArgumentParser(
    #     description="Script for a simple test loop evaluating a network on the test dataset")
    # parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    # args = parser.parse_args()


    # exp_path = args.EXP_PATH

    # Code for hard coding experiment into script
    exp_path = 'logdir/classifier/synth_normalnet'

    model_path = os.path.join(base_path, exp_path)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config)
