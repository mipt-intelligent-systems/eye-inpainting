import numpy as np
import tensorflow as tf
import cv2
import tqdm
from src.network import Network
from src.autoencoder import Autoencoder, EYE_SIZE
from src.datasets import get_full_dataset
from src.utils.paths import PATH_DATA
from os.path import join

IMAGE_SIZE = 128
LOCAL_SIZE = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
PRETRAIN_EPOCH = 10

PATH_CELEB_ALIGN_IMAGES = join(PATH_DATA, 'celeb_id_aligned')


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


def train(train_size):
    sess = tf.Session()

    is_training = tf.placeholder(tf.bool, [])
    is_left_eye = tf.placeholder(tf.bool, [])
    x_autoencoder = tf.placeholder(tf.float32, [BATCH_SIZE, EYE_SIZE, EYE_SIZE, 3])
    autoencoder = Autoencoder(x_autoencoder, is_left_eye, is_training, BATCH_SIZE)


    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    reference = tf.placeholder(tf.float32, [BATCH_SIZE, 256])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 8])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    local_x_right = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    local_completion_right = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    reference_left = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    reference_right = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    model = Network(x, mask, points, local_x, local_x_right,
                    global_completion, local_completion, local_completion_right,
                    reference_left, reference_right,
                    is_training, batch_size=BATCH_SIZE, autoencoder=autoencoder)
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)
    g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    ref_train_op = opt.minimize(model.reference_loss, global_step=global_step, var_list=model.g_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if tf.train.get_checkpoint_state('./backup'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest')

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
    saver.restore(sess, './backup_autoencoder/latest')


    train_generator, test_generator = get_full_dataset(PATH_CELEB_ALIGN_IMAGES)
    
    step_num = int(train_size / BATCH_SIZE)

    while True:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        g_loss_value = 0
        ref_loss_value = 0
        for i, (x_batch, mask_batch, points_batch, reference_left_batch, reference_right_batch) in tqdm.tqdm(enumerate(train_generator(BATCH_SIZE)), total=step_num):
            if i == step_num:
                break
            _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, mask: mask_batch, reference_left: reference_left_batch, reference_right: reference_right_batch, points: points_batch, is_training: True})
            g_loss_value += g_loss

            all_new_points = []
            for i in range(BATCH_SIZE):
                new_points = []
                point = points_batch[i].reshape(2, 4)[0]
                x1, y1, x2, y2 = point_to_coords(point)
                point = points_batch[i].reshape(2, 4)[1]
                new_points.extend([x1, y1, x2, y2])
                x1, y1, x2, y2 = point_to_coords(point)
                new_points.extend([x1, y1, x2, y2])
                all_new_points.append(np.array(new_points, dtype=np.uint8))
            points_batch = np.array(all_new_points)

            _, ref_loss = sess.run([ref_train_op, model.reference_loss],\
                feed_dict={x: x_batch, mask: mask_batch, reference_left: reference_left_batch,\
                reference_right: reference_right_batch, points: points_batch, is_training: True})

            ref_loss_value += ref_loss

        print('Completion loss: {}'.format(g_loss_value))
        print('Reference loss: {}'.format(ref_loss_value))

        x_batch, mask_batch, _, reference_left_batch, reference_right_batch = next(test_generator(BATCH_SIZE))
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, reference_left: reference_left_batch, reference_right: reference_right_batch, is_training: False})
        sample = np.array((-completion[0] + 1) * 127.5, dtype=np.uint8)
        cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

        saver = tf.train.Saver(model.g_variables)
        saver.save(sess, './backup/latest', write_meta_graph=False)


if __name__ == '__main__':
    train()
