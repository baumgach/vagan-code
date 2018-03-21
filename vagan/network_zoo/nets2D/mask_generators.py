import tensorflow as tf
from tfwrapper import layers

def unet_16_2D_bn(x, training, scope_name='generator'):

    n_ch_0 = 16

    with tf.variable_scope(scope_name):

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=n_ch_0, training=training)
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=n_ch_0, training=training)
        pool1 = layers.maxpool2D_layer(conv1_2)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=n_ch_0*2, training=training)
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=n_ch_0*2, training=training)
        pool2 = layers.maxpool2D_layer(conv2_2)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=n_ch_0*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=n_ch_0*4, training=training)
        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=n_ch_0*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=n_ch_0*8, training=training)

        upconv3 = layers.deconv2D_layer_bn(conv4_2, name='upconv3', num_filters=n_ch_0, training=training)
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=-1)

        conv5_1 = layers.conv2D_layer_bn(concat3, 'conv5_1', num_filters=n_ch_0*4, training=training)

        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=n_ch_0*4, training=training)

        upconv2 = layers.deconv2D_layer_bn(conv5_2, name='upconv2', num_filters=n_ch_0, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=-1)

        conv6_1 = layers.conv2D_layer_bn(concat2, 'conv6_1', num_filters=n_ch_0*2, training=training)
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=n_ch_0*2, training=training)

        upconv1 = layers.deconv2D_layer_bn(conv6_2, name='upconv1', num_filters=n_ch_0, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=-1)

        conv8_1 = layers.conv2D_layer_bn(concat1, 'conv8_1', num_filters=n_ch_0, training=training)
        conv8_2 = layers.conv2D_layer(conv8_1, 'conv8_2', num_filters=1, activation=tf.identity)

    return conv8_2
