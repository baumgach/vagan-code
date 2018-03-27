
import tensorflow as tf
from tfwrapper import layers


def C3D_fcn_16(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool3D_layer(conv1_1)

        conv2_1 = layers.conv3D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool3D_layer(conv2_1)

        conv3_1 = layers.conv3D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv3D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer(pool3, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv3D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool3D_layer(conv4_2)

        conv5_1 = layers.conv3D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv3D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv3D_layer(conv5_2, 'convD_1', num_filters=256)
        convD_2 = layers.conv3D_layer(convD_1,
                                         'convD_2',
                                         num_filters=1,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        logits = layers.averagepool3D_layer(convD_2, name='diagnosis_avg')


    return logits
