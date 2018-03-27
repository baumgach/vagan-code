# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers


def normalnet2D(x, nlabels, training, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        init_filters = 32

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=init_filters, training=training)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=init_filters*2, training=training)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=init_filters*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=init_filters*4, training=training)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=init_filters*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=init_filters*8, training=training)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=init_filters*16, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=init_filters*16, training=training)

        convD_1 = layers.conv2D_layer_bn(conv5_2, 'convD_1', num_filters=init_filters*16, training=training)

        dense1 = layers.dense_layer_bn(convD_1, 'dense1', hidden_units=init_filters*16, training=training)

        logits = layers.dense_layer_bn(dense1, 'dense2', hidden_units=nlabels, training=training, activation=tf.identity)


    return logits


def rebuttalnet2D(x, nlabels, training, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        init_filters = 32

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=init_filters, training=training)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=init_filters*2, training=training)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=init_filters*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=init_filters*4, training=training)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=init_filters*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=init_filters*8, training=training)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=init_filters*16, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=init_filters*16, training=training)

        convD_1 = layers.conv2D_layer_bn(conv5_2, 'convD_1', num_filters=init_filters*16, training=training)

        avg_pool = layers.averagepool2D_layer(convD_1, name='avg_pool')

        logits = layers.dense_layer_bn(avg_pool, 'dense2', hidden_units=nlabels, training=training, activation=tf.identity)


    return logits


def VGG16(x, nlabels, training, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        init_filters = 64

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=init_filters, training=training)
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=init_filters, training=training)

        pool1 = layers.maxpool2D_layer(conv1_2)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=init_filters*2, training=training)
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=init_filters*2, training=training)

        pool2 = layers.maxpool2D_layer(conv2_2)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=init_filters*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=init_filters*4, training=training)
        conv3_3 = layers.conv2D_layer_bn(conv3_2, 'conv3_3', num_filters=init_filters*4, training=training)

        pool3 = layers.maxpool2D_layer(conv3_3)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=init_filters*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=init_filters*8, training=training)
        conv4_3 = layers.conv2D_layer_bn(conv4_2, 'conv4_3', num_filters=init_filters*8, training=training)

        pool4 = layers.maxpool2D_layer(conv4_3)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=init_filters*8, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=init_filters*8, training=training)
        conv5_3 = layers.conv2D_layer_bn(conv5_2, 'conv5_3', num_filters=init_filters*8, training=training)

        pool5 = layers.maxpool2D_layer(conv5_3)

        dense1 = layers.dense_layer_bn(pool5, 'dense1', hidden_units=init_filters*64, training=training)
        dense2 = layers.dense_layer_bn(dense1, 'dense2', hidden_units=init_filters*64, training=training)

        logits = layers.dense_layer_bn(dense2, 'dense3', hidden_units=nlabels, training=training, activation=tf.identity)


    return logits


def normalnet_deeper2D(x, nlabels, training, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        init_filters = 32

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=init_filters, training=training)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=init_filters*2, training=training)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=init_filters*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=init_filters*4, training=training)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=init_filters*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=init_filters*8, training=training)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=init_filters*16, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=init_filters*16, training=training)

        pool5 = layers.maxpool2D_layer(conv5_2)

        conv6_1 = layers.conv2D_layer_bn(pool5, 'conv6_1', num_filters=init_filters*16, training=training)
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=init_filters*16, training=training)

        dense1 = layers.dense_layer_bn(conv6_2, 'dense1', hidden_units=init_filters*16, training=training)

        logits = layers.dense_layer_bn(dense1, 'dense2', hidden_units=nlabels, training=training, activation=tf.identity)


    return logits


def CAM_net2D(x, nlabels, training, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        init_filters = 32

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=init_filters, training=training)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=init_filters*2, training=training)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=init_filters*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=init_filters*4, training=training)

        conv4_1 = layers.conv2D_layer_bn(conv3_2, 'conv4_1', num_filters=init_filters*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=init_filters*8, training=training)

        conv5_1 = layers.conv2D_layer_bn(conv4_2, 'conv5_1', num_filters=init_filters*16, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=init_filters*16, training=training)

        convD_1 = layers.conv2D_layer_bn(conv5_2, 'feature_maps', num_filters=init_filters*16, training=training)

        fm_averages = layers.averagepool2D_layer(convD_1, name='fm_averages')

        logits = layers.dense_layer(fm_averages, 'weight_layer', hidden_units=nlabels, activation=tf.identity, add_bias=False)

    return logits