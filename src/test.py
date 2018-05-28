import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from src.network import Network
from src.utils.paths import PATH_DATA, PATH_WEIGHTS, PATH_OUTPUT
from src.datasets import get_full_dataset
from os.path import join

IMAGE_SIZE = 128
LOCAL_SIZE = 32
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 16
PRETRAIN_EPOCH = 100

PATH_CELEB_ALIGN_IMAGES = join(PATH_DATA, 'celeb_id_aligned')

weights_path = join(PATH_WEIGHTS, 'latest')


def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    reference = tf.placeholder(tf.float32, [BATCH_SIZE, 256])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 8])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, reference, points, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, weights_path)

    _, test_generator = get_full_dataset(PATH_CELEB_ALIGN_IMAGES)

    cnt = 0
    for i, (X_batch, mask_batch, _, reference_batch) in tqdm.tqdm(enumerate(test_generator(BATCH_SIZE))):
        
        completion = sess.run(model.completion, feed_dict={x: X_batch, mask: mask_batch, reference: reference_batch, is_training: False})
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = X_batch[i]
            raw = np.array((-raw + 1) * 127.5, dtype=np.uint8)
            masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
            img = completion[i]
            img = np.array((-img + 1) * 127.5, dtype=np.uint8)
            dst = join(PATH_OUTPUT, '{}.jpg'.format("{0:06d}".format(cnt)))
            output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst)
            

def get_mask(input_images):
    # восстанавливаем из картинки её маску
    mask = []
    print(input_images[0])
    for i in range(BATCH_SIZE):
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        for y in range(IMAGE_SIZE):
            for x in range(IMAGE_SIZE):
                if input_images[i, y, x, 0] == -1 and input_images[i, y, x, 1] == -1 and input_images[i, y, x, 2] == -1:
                    m[y, x] = 1
        mask.append(m)
    return np.array(mask)
    

def output_image(images, dst):
    fig = plt.figure()
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()


if __name__ == '__main__':
    test()
