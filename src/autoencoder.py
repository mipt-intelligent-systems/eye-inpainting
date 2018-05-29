import tensorflow as tf
from src.layer import conv_layer, batch_normalize, flatten_layer, full_connection_layer, deconv_layer

EYE_SIZE = 16


class Autoencoder:
    def __init__(self, x, is_training, batch_size):
        self.input = x
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
                    x = tf.reshape(x, [self.batch_size, 1, 1, 256])
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


if __name__ == '__main__':
    import tqdm
    import numpy as np
    from os.path import join
    from src.datasets import get_full_dataset
    from src.utils.paths import PATH_DATA

    PATH_CELEB_ALIGN_IMAGES = join(PATH_DATA, 'celeb_id_aligned')

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3


    def train(train_size=97453):
        x = tf.placeholder(tf.float32, [BATCH_SIZE * 2, EYE_SIZE, EYE_SIZE, 3])
        is_training = tf.placeholder(tf.bool, [])

        model = Autoencoder(x, is_training, BATCH_SIZE * 2)
        sess = tf.Session()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)

        opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = opt.minimize(model.loss, global_step=global_step, var_list=model.variables)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        if tf.train.get_checkpoint_state('./backup_autoencoder'):
            saver = tf.train.Saver()
            saver.restore(sess, './backup_autoencoder/latest')

        train_generator, test_generator = get_full_dataset(PATH_CELEB_ALIGN_IMAGES)

        step_num = int(train_size / BATCH_SIZE)

        while True:
            sess.run(tf.assign(epoch, tf.add(epoch, 1)))
            print('epoch: {}'.format(sess.run(epoch)))

            # np.random.shuffle(x_train)
            train_loss = 0
            for i, (x_batch, _, points_batch, _) in tqdm.tqdm(enumerate(train_generator(BATCH_SIZE)), total=step_num):
                if i == step_num:
                    break
                x_batch_new = []
                for image, points in zip(x_batch, points_batch):
                    fx1, fy1, fx2, fy2 = points[0], points[1], points[2], points[3]
                    sx1, sy1, sx2, sy2 = points[4], points[5], points[6], points[7]
                    first_eye = image[fy1:fy1 + EYE_SIZE, fx1:fx1 + EYE_SIZE, :]
                    second_eye = image[sy1:sy1 + EYE_SIZE, sx1:sx1 + EYE_SIZE, :]
                    x_batch_new.append(first_eye)
                    x_batch_new.append(second_eye)
                _, loss = sess.run([train_op, model.loss],
                                   feed_dict={x: np.array(x_batch_new), is_training: True})
                train_loss += loss

            print('Loss: {}'.format(train_loss))

            saver = tf.train.Saver()
            saver.save(sess, './backup_autoencoder/latest', write_meta_graph=False)


    train()
