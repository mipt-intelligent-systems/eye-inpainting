import tensorflow as tf
from src.layer import conv_layer, batch_normalize, dilated_conv_layer, deconv_layer
from src.layer import flatten_layer, full_connection_layer


def eye_feature_extractor(eye_image, is_left_eye, autoencoder):
    feed_dict = {autoencoder.inputs['image']: [eye_image],
                 autoencoder.inputs['is_left_eye']: is_left_eye,
                 autoencoder.inputs['is_training']: False}
    return tf.run(autoencoder.encoded, feed_dict)[0]


def extract_features(image, points, autoencoder, eye_size=16):
    points = tf.clip_by_value(points, 0, 127)
    fx1, fy1, fx2, fy2 = points[0], points[1], points[2], points[3]
    sx1, sy1, sx2, sy2 = points[4], points[5], points[6], points[7]
    # tf.Tensor supports slicing
    first_eye = image[fy1:fy1 + eye_size, fx1:fx1 + eye_size, :]
    second_eye = image[sy1:sy1 + eye_size, sx1:sx1 + eye_size, :]
    first_features = eye_feature_extractor(first_eye, True, autoencoder)
    second_features = eye_feature_extractor(second_eye, False, autoencoder)
    return tf.concat([first_features, second_features], 0)


def extract_features_from_batch(image_batch, points_batch, autoencoder):
    return tf.map_fn(lambda x: extract_features(x[0], x[1], autoencoder),
                     (image_batch, points_batch), dtype=tf.float32)

def extract_features_from_eye_batches(left_batch, rigth_batch, autoencoder):
    left_result = tf.run(autoencoder.encoded, {autoencoder.inputs['image']: left_batch,
                 autoencoder.inputs['is_left_eye']: True,
                 autoencoder.inputs['is_training']: False})
    right_result = tf.run(autoencoder.encoded, {autoencoder.inputs['image']: rigth_batch,
                 autoencoder.inputs['is_left_eye']: False,
                 autoencoder.inputs['is_training']: False})
    return tf.concat([left_result, right_result], -1)


class Network:
    def __init__(self, x, mask, reference, points, local_x, local_x_right,
                 global_completion, local_completion, local_completion_right,
                 is_training, batch_size, autoencoder):
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.imitation = self.generator(x * (1 - mask), is_training)
        self.completion = self.imitation * mask + x * (1 - mask)
        self.real = self.discriminator(x, local_x, local_x_right, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, local_completion_right, reuse=True)
        self.g_loss = self.calc_g_loss(x, self.completion)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.reference_loss = self.calc_reference_loss(reference, self.completion, points)
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
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

    def calc_reference_loss(self, reference, left_batch, right_batch):
        result_reference = extract_features_from_eye_batches(left_batch, right_batch, self.autoencoder)
        loss = tf.nn.l2_loss(result_reference - reference)
        return tf.reduce_mean(loss)

    def calc_d_loss(self, real, fake):
        alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake) * alpha
