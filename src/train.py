import numpy as np
import tensorflow as tf
import cv2
import tqdm
from src.network import Network
from src.datasets import load

IMAGE_SIZE = 128
LOCAL_SIZE = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
PRETRAIN_EPOCH = 1

def train():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if tf.train.get_checkpoint_state('./backup'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest')

    x_train, masks_train, points_train, x_test, masks_test, points_test = load()
    x_train = np.transpose(x_train, (0, 2, 3, 1)) 
    masks_train = np.transpose(masks_train, (0, 2, 3, 1)) 
    x_test = np.transpose(x_test, (0, 2, 3, 1)) 
    masks_test = np.transpose(masks_test, (0, 2, 3, 1)) 
    
    step_num = int(len(x_train) / BATCH_SIZE)

    while True:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        # np.random.shuffle(x_train)

        # Completion 
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            g_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                
                mask_batch, points_batch = masks_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], points_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

            print('Completion loss: {}'.format(g_loss_value))

            np.random.shuffle(x_test) 
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))


            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            if sess.run(epoch) == PRETRAIN_EPOCH:
                saver.save(sess, './backup/pretrained', write_meta_graph=False)


        # Discrimitation
        else:
            g_loss_value = 0
            d_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                mask_batch, points_batch = masks_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], points_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

                local_x_batch = []
                local_completion_batch = []
                for i in range(BATCH_SIZE):
                    point = points_batch[i].reshape(2, 4)[0]
                    x1, y1, x2, y2 = point[0], point[1], point[0] + LOCAL_SIZE, point[1] + LOCAL_SIZE
                    local_x_batch.append(x_batch[i][y1:y2, x1:x2, :])
                    local_completion_batch.append(completion[i][y1:y2, x1:x2, :])
                local_x_batch = np.array(local_x_batch)
                local_completion_batch = np.array(local_completion_batch)

                _, d_loss = sess.run(
                    [d_train_op, model.d_loss], 
                    feed_dict={x: x_batch, mask: mask_batch, local_x: local_x_batch, global_completion: completion, local_completion: local_completion_batch, is_training: True})
                d_loss_value += d_loss

            print('Completion loss: {}'.format(g_loss_value))
            print('Discriminator loss: {}'.format(d_loss_value))

            np.random.shuffle(x_test) 
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)


if __name__ == '__main__':
    train()
    
