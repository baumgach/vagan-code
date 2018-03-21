# Code for displaying visual feature attribution maps for a trained VA-GAN model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse

import config.system as sys_config
from vagan.model_vagan import vagan

#######################################################################

if not sys_config.running_on_gpu_host:
    import matplotlib.pyplot as plt


def plot_slices(ad_in, morphed, mask):

    if ad_in.ndim == 5:

        xslice = 40
        yslice = 83
        zslice = 56

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.squeeze(ad_in[0, :, :, zslice, :]), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))
        ax2 = fig.add_subplot(132)
        ax2.imshow(np.squeeze(morphed[0, :, :, zslice, :]), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))

        difference = -np.squeeze(mask[0, :, :, zslice, :])
        ax3 = fig.add_subplot(133)
        ax3.imshow(difference, cmap='gray')

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.transpose(np.squeeze(ad_in[0, xslice, :, ::-1, :])), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))
        ax2 = fig.add_subplot(132)
        ax2.imshow(np.transpose(np.squeeze(morphed[0, xslice, :, ::-1, :])), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))

        difference = -np.transpose(np.squeeze(mask[0, xslice, :, ::-1, :]))
        ax3 = fig.add_subplot(133)
        ax3.imshow(difference, cmap='gray')

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.transpose(np.squeeze(ad_in[0, :, yslice, ::-1, :])), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))
        ax2 = fig.add_subplot(132)
        ax2.imshow(np.transpose(np.squeeze(morphed[0, :, yslice, ::-1, :])), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))

        difference = -np.transpose(np.squeeze(mask[0, :, yslice, ::-1, :]))
        ax3 = fig.add_subplot(133)
        ax3.imshow(difference, cmap='gray')

        plt.show()

    elif ad_in.ndim == 4:

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.squeeze(ad_in[0, :, :, :]), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))
        ax2 = fig.add_subplot(132)
        ax2.imshow(np.squeeze(morphed[0, :, :, :]), cmap='gray', vmin=np.min(ad_in), vmax=np.max(ad_in))

        difference = -np.squeeze(mask[0, :, :, :])
        ax3 = fig.add_subplot(133)
        ax3.imshow(difference)

        plt.show()

    else:

        raise ValueError('Invalid number of dimensions for plot function: %d. Possible values are 4 or 5' % ad_in.ndim)



def main(exp_config):

    # GAN Settings
    gan_log_dir = os.path.join(sys_config.log_root, 'gan', exp_config.experiment_name)

    # Data Settings
    if exp_config.data_identifier == 'synthetic':
        from data.synthetic_data import synthetic_data as data_loader
    elif exp_config.data_identifier == 'adni':
        from data.adni_data import adni_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % exp_config.data_identifier)

    data = data_loader(exp_config)

    # Make and restore vagan model
    vagan_model = vagan(exp_config=exp_config, data=data)
    vagan_model.load_weights(gan_log_dir, type='latest')

    # Run predictions in an endless loop
    sampler_AD = lambda bs: data.testAD.next_batch(bs)[0]
    while True:

        ad_in = sampler_AD(1)
        mask = vagan_model.predict_mask(ad_in)

        morphed = ad_in + mask
        if exp_config.use_tanh:
            morphed = np.tanh(morphed)

        if not sys_config.running_on_gpu_host:
            if exp_config.use_tanh:
                plot_slices(ad_in, morphed, morphed - ad_in)
            else:
                plot_slices(ad_in, morphed, mask)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a nets2D network on slices from the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    init_iteration = main(exp_config=exp_config)
