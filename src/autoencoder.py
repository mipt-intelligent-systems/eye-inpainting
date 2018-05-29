import tensorflow as tf
from src.layer import conv_layer, batch_normalize, flatten_layer, full_connection_layer, deconv_layer


class Autoencoder:
    def __init__(self, x, is_training, batch_size):
        self.batch_size = batch_size
        self.encoded = self.encoder(x, is_training)
        self.decoded_left = self.decoder(self.encoded, is_training, 'left')
        self.decoded_right = self.decoder(self.encoded, is_training, 'right')
        self.loss_left = self.calc_loss(x, self.decoded_left)
        self.loss_right = self.calc_loss(x, self.decoded_right)
        self.loss = tf.add(self.loss_left, self.loss_right)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')

    def encoder(self, x, is_training):
        with tf.variable_scope('autoencoder'):
            with tf.variable_scope('encoder'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 16], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [3, 3, 16, 32], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [3, 3, 32, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [3, 3, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [3, 3, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 256)
        return x

    def decoder(self, x, is_training, name_prefix):
        with tf.variable_scope('autoencoder'):
            with tf.variable_scope(f'decoder_{name_prefix}'):
                with tf.variable_scope('fc'):
                    x = full_connection_layer(x, 256)
                    x = tf.reshape(x, [-1, 256, 1, 1])
                with tf.variable_scope('deconv1'):
                    x = deconv_layer(x, [3, 3, 128, 256], [self.batch_size, 2, 2, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('deconv2'):
                    x = deconv_layer(x, [3, 3, 64, 128], [self.batch_size, 4, 4, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('deconv3'):
                    x = deconv_layer(x, [3, 3, 32, 64], [self.batch_size, 8, 8, 32], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('deconv4'):
                    x = deconv_layer(x, [3, 3, 16, 32], [self.batch_size, 16, 16, 16], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 16, 3], 1)
                    x = tf.nn.tanh(x)
            return x

    def calc_loss(self, x_original, x_decoded):
        loss = tf.nn.l2_loss(x_original - x_decoded)
        return tf.reduce_mean(loss)
