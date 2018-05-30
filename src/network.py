import tensorflow as tf
import numpy as np
from src.layer import conv_layer, batch_normalize, dilated_conv_layer, deconv_layer
from src.layer import flatten_layer, full_connection_layer

IMAGE_SIZE = 128
LOCAL_SIZE = 32

def extract_features_from_eye_batches(left_batch, right_batch, autoencoder):
    left_result = autoencoder.encoder(left_batch, tf.cast(False, tf.bool), reuse=True)
    right_result = autoencoder.encoder(right_batch, tf.cast(False, tf.bool), reuse=True)
    return tf.concat([left_result, right_result], -1)

def get_sliced_eyes(completion, points):
    fx1, fy1, fx2, fy2 = points[0], points[1], points[0] + LOCAL_SIZE, points[1] + LOCAL_SIZE
    sx1, sy1, sx2, sy2 = points[4], points[5], points[4] + LOCAL_SIZE, points[5] + LOCAL_SIZE
    first_eye = completion[fy1:fy2, fx1:fx2, :]
    second_eye = completion[sy1:sy2, sx1:sx2, :]
    first_eye.set_shape([LOCAL_SIZE, LOCAL_SIZE, 3])
    second_eye.set_shape([LOCAL_SIZE, LOCAL_SIZE, 3])
    return first_eye, second_eye

class Network:
    def __init__(self, x, mask, points, local_x, local_x_right,
                 global_completion, local_completion, local_completion_right,
                 reference_left, reference_right,
                 is_training, batch_size, autoencoder):
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.left_ref = reference_left
        self.right_ref = reference_right
        self.reference = extract_features_from_eye_batches(reference_left, reference_right, autoencoder)
        self.imitation = self.generator(x * (1 - mask), self.reference, is_training)
        self.completion = self.imitation * mask + x * (1 - mask)
        self.real = self.discriminator(x, local_x, local_x_right, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, local_completion_right, reuse=True)
        self.g_loss = self.calc_g_loss(x, self.completion)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.reference_loss = self.calc_reference_loss(points, self.completion)
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    def generator(self, x, ref, is_training):
        with tf.variable_scope('generator'):
            # encoder
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [5, 5, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated1'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated2'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 4)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated3'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 8)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated4'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 16)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('add_reference'):
                ref_compact = full_connection_layer(ref, 256)
                reshaped_reference = tf.reshape(tf.tile(ref_compact, (1, 32 * 32)), [self.batch_size, 32, 32, 256])
                x = tf.concat([x, reshaped_reference], -1)
            with tf.variable_scope('conv_add_reference'):
                x = conv_layer(x, [3, 3, 512, 256], 1)
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 64, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 128, 128, 64], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv10'):
                x = conv_layer(x, [3, 3, 64, 32], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                x = tf.nn.tanh(x)
        return x

        return x

    def discriminator(self, global_x, local_x, local_x_right, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_dicriminator_right(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local_right'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            local_right_output = local_dicriminator_right(local_x_right)
            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output, local_right_output), 1)
                output = full_connection_layer(output, 1)
               
        return output

    def calc_g_loss(self, x, completion):
        loss = tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)

    def calc_reference_loss(self, points, image):
        sliced_eyes = tf.map_fn(lambda x: get_sliced_eyes(x[0], x[1]), (image, points), dtype=(tf.float32, tf.float32))
        result_reference = extract_features_from_eye_batches(sliced_eyes[0], sliced_eyes[1], self.autoencoder)
        loss = tf.nn.l2_loss(result_reference - self.reference)
        return tf.reduce_mean(loss)

    def calc_d_loss(self, real, fake):
        alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake) * alpha
