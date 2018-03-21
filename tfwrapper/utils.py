# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np
import math
import glob
import os

def flatten(tensor):
    '''
    Flatten the last N-1 dimensions of a tensor only keeping the first one, which is typically 
    equal to the number of batches. 
    Example: A tensor of shape [10, 200, 200, 32] becomes [10, 1280000] 
    '''
    rhs_dim = get_rhs_dim(tensor)
    return tf.reshape(tensor, [-1, rhs_dim])

def get_rhs_dim(tensor):
    '''
    Get the multiplied dimensions of the last N-1 dimensions of a tensor. 
    I.e. an input tensor with shape [10, 200, 200, 32] leads to an output of 1280000 
    '''
    shape = tensor.get_shape().as_list()
    return np.prod(shape[1:])


def put_kernels_on_grid(images, batch_size, pad=1, min_int=None, max_int=None):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
      images:            [batch_size, X, Y, channels] 
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    pass
                return (i, int(n / i))

    # (grid_Y, grid_X) = factorization(images.get_shape()[0].value)
    # print('grid: %d = (%d, %d)' % (images.get_shape()[0].value, grid_Y, grid_X))

    (grid_Y, grid_X) = factorization(batch_size)
    # print('grid: %d = (%d, %d)' % (batch_size, grid_Y, grid_X))

    if not min_int:
        x_min = tf.reduce_min(images)
    else:
        x_min = min_int

    if not max_int:
        x_max = tf.reduce_max(images)
    else:
        x_max = max_int

    images = (images - x_min) / (x_max - x_min)
    images = 255.0 * images

    # pad X and Y
    x = tf.pad(images, tf.constant([[0, 0], [pad, pad], [pad, pad],[0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = images.get_shape().as_list()[1] + 2 * pad
    X = images.get_shape().as_list()[2] + 2 * pad

    channels = images.get_shape()[3]

    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))

    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # Transpose the image again
    x = tf.transpose(x, (0, 2, 1, 3))

    return x


from tensorflow.python import pywrap_tensorflow
def print_tensornames_in_checkpoint_file(file_name):
    """ 
    """

    reader = pywrap_tensorflow.NewCheckpointReader(file_name)

    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print(" - tensor_name: ", key)
        #print(reader.get_tensor(key))

def get_checkpoint_weights(file_name):
    """ 
    """
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    return {n: reader.get_tensor(n) for n in reader.get_variable_to_shape_map()}


def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    if len(iteration_nums) == 0:
        return False

    latest_iteration = np.max(iteration_nums)
    return os.path.join(folder, name + '-' + str(latest_iteration))


def total_variation(images, beta=1.0, mode='2D'):

    ndims = images.get_shape().ndims

    if mode == '3D':

        if ndims == 4:
            # The input is a single image with shape [height, width, channels].
            pixel_dif1 = images[1:, :-1, :-1, :] - images[:-1, :-1, :-1, :]
            pixel_dif2 = images[:-1, 1:, :-1, :] - images[:-1, :-1, :-1, :]
            pixel_dif3 = images[:-1, :-1, 1:, :] - images[:-1, :-1, :-1, :]
            sum_axis = None
        elif ndims == 5:
            pixel_dif1 = images[:, 1:, :-1, :-1, :] - images[:, :-1, :-1, :-1, :]
            pixel_dif2 = images[:, :-1, 1:, :-1, :] - images[:, :-1, :-1, :-1, :]
            pixel_dif3 = images[:, :-1, :-1, 1:, :] - images[:, :-1, :-1, :-1, :]
            sum_axis = [1, 2, 3, 4]
        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        a = tf.square(tf.abs(pixel_dif1))
        b = tf.square(tf.abs(pixel_dif2))
        c = tf.square(tf.abs(pixel_dif3))

        tot_var = tf.reduce_sum(tf.pow(a + b + c, beta / 2.0), axis=sum_axis)

        return tot_var

    if mode == '2D':
        if ndims == 3:
            # The input is a single image with shape [height, width, channels].
            pixel_dif1 = images[1:, :-1, :] - images[:-1, :-1, :]
            pixel_dif2 = images[:-1, 1:, :] - images[:-1, :-1, :]
            sum_axis = None
        elif ndims == 4:
            pixel_dif1 = images[:, 1:, :-1, :] - images[:, :-1, :-1, :]
            pixel_dif2 = images[:, :-1, 1:, :] - images[:, :-1, :-1, :]
            sum_axis = [1, 2, 3]
        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        a = tf.square(tf.abs(pixel_dif1))
        b = tf.square(tf.abs(pixel_dif2))

        tot_var = tf.reduce_sum(tf.pow(a + b, beta / 2.0), axis=sum_axis)

        return tot_var

    else:

        raise ValueError('Mode must be 2D or 3D')