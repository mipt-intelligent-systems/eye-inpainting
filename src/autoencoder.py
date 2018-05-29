import tensorflow as tf
from src.layer import conv_layer, batch_normalize, flatten_layer, full_connection_layer, deconv_layer
import tqdm
import numpy as np
from os.path import join
from src.datasets import get_initial_train_dataset
from src.utils.paths import PATH_DATA

EYE_SIZE = 32
LOCAL_SIZE = 32
IMAGE_SIZE = 128

def point_to_coords(point):
    point[0] = max(0, point[0])
    point[1] = max(0, point[1])
    x1, y1, x2, y2 = point[0], point[1], point[0] + LOCAL_SIZE, point[1] + LOCAL_SIZE
    if x2 > IMAGE_SIZE:
        x2 = IMAGE_SIZE
        x1 = x2 - LOCAL_SIZE
    if y2 > IMAGE_SIZE:
        y2 = IMAGE_SIZE
        y1 = y2 - LOCAL_SIZE
    return x1, y1, x2, y2

class Autoencoder:
    def __init__(self, x, is_left_eye, is_training, batch_size):
        self.inputs = {
            'image': x,
            'is_left_eye': is_left_eye,
            'is_training': is_training
        }

        self.batch_size = batch_size
        self.encoded = self.encoder(x, is_training)
        self.decoded_left = self.decoder(self.encoded, is_training, 'left')
        self.decoded_right = self.decoder(self.encoded, is_training, 'right')
        self.loss_left = self.calc_loss(x, self.decoded_left)
        self.loss_right = self.calc_loss(x, self.decoded_right)
        self.loss = tf.cond(is_left_eye, lambda: self.loss_left, lambda: self.loss_right)
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
                with tf.variable_scope('conv6'):
                    x = conv_layer(x, [3, 3, 256, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 256)
        return x

    def decoder(self, x, is_training, name_prefix):
        with tf.variable_scope('autoencoder'):
            with tf.variable_scope('decoder_%s' % (name_prefix)):
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
                with tf.variable_scope('deconv5'):
                    x = deconv_layer(x, [3, 3, 16, 16], [self.batch_size, 32, 32, 16], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 16, 3], 1)
                    x = tf.nn.tanh(x)
            return x

    def calc_loss(self, x_original, x_decoded):
        loss = tf.nn.l2_loss(x_original - x_decoded)
        return tf.reduce_mean(loss)

PATH_CELEB_ALIGN_IMAGES = join(PATH_DATA, 'celeb_id_aligned')

BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def train(train_size=97453):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, EYE_SIZE, EYE_SIZE, 3])
    is_left_eye = tf.placeholder(tf.bool, [])
    is_training = tf.placeholder(tf.bool, [])

    model = Autoencoder(x, is_left_eye, is_training, BATCH_SIZE)
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

    train_generator = get_initial_train_dataset()

    step_num = int(train_size / BATCH_SIZE)

    while True:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        # np.random.shuffle(x_train)
        train_loss = 0
        for i, (x_batch, _, points_batch) in tqdm.tqdm(enumerate(train_generator(BATCH_SIZE)), total=step_num):
            if i == step_num:
                break
            x_batch_left, x_batch_right = [], []
            for image, points in zip(x_batch, points_batch):
                
                fx1, fy1, fx2, fy2 = point_to_coords(points.reshape(2, 4)[0])
                sx1, sy1, sx2, sy2 = point_to_coords(points.reshape(2, 4)[1])
                first_eye = image[fy1:fy1 + EYE_SIZE, fx1:fx1 + EYE_SIZE, :]
                second_eye = image[sy1:sy1 + EYE_SIZE, sx1:sx1 + EYE_SIZE, :]
                x_batch_left.append(first_eye)
                x_batch_right.append(second_eye)
            _, loss_left = sess.run([train_op, model.loss],
                                    feed_dict={x: np.array(x_batch_left), is_left_eye: True, is_training: True})
            _, loss_right = sess.run([train_op, model.loss],
                                        feed_dict={x: np.array(x_batch_right), is_left_eye: False, is_training: True})
            train_loss += loss_left + loss_right

        print('Loss: {}'.format(train_loss))

        saver = tf.train.Saver()
        saver.save(sess, './backup_autoencoder/latest', write_meta_graph=False)

if __name__ == '__main__':
    train()
