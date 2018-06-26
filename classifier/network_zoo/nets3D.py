# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import tensorflow as tf
from tfwrapper import layers


def FCN_32_bn(images, training, nlabels, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, training=training)

        pool1 = layers.maxpool3D_layer(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, training=training)

        pool2 = layers.maxpool3D_layer(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training)

        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training)

        pool4 = layers.maxpool3D_layer(conv4_2)

        conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training)
        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training)

        convD_1 = layers.conv3D_layer_bn(conv5_2, 'convD_1', num_filters=256, training=training)
        convD_2 = layers.conv3D_layer_bn(convD_1,
                                         'convD_2',
                                         num_filters=nlabels,
                                         training=training,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        diag_logits = layers.averagepool3D_layer(convD_2, name='diagnosis_avg')

    return diag_logits



def allconv_bn(images, training, nlabels, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, training=training, strides=(2,2,2))

        conv2_1 = layers.conv3D_layer_bn(conv1_1, 'conv2_1', num_filters=64, training=training, strides=(2,2,2))

        conv3_1 = layers.conv3D_layer_bn(conv2_1, 'conv3_1', num_filters=128, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training, strides=(2,2,2))

        conv4_1 = layers.conv3D_layer_bn(conv3_2, 'conv4_1', num_filters=256, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training, strides=(2,2,2))

        conv5_1 = layers.conv3D_layer_bn(conv4_2, 'conv5_1', num_filters=256, training=training)
        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training)

        convD_1 = layers.conv3D_layer_bn(conv5_2, 'convD_1', num_filters=256, training=training)
        convD_2 = layers.conv3D_layer_bn(convD_1,
                                         'convD_2',
                                         num_filters=nlabels,
                                         training=training,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        diag_logits = layers.averagepool3D_layer(convD_2, name='diagnosis_avg')

    return diag_logits


def C3D_32_bn(images, training, nlabels, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, training=training)

        pool1 = layers.maxpool3D_layer(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, training=training)

        pool2 = layers.maxpool3D_layer(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training)

        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training)

        pool4 = layers.maxpool3D_layer(conv4_2)

        dense1 = layers.dense_layer_bn(pool4, 'dense1', hidden_units=512, training=training)
        diag_logits = layers.dense_layer_bn(dense1, 'diag_logits', hidden_units=nlabels, activation=tf.identity, training=training)

    return diag_logits
