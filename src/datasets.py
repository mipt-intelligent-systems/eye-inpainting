import os
import numpy as np
from src.utils.paths import PROCESSED_NPY_DATA, PATH_DATA
from src.utils.io import read_image, draw_rectangle, make_input_image, get_rects
import json
from os.path import join
from tqdm import tqdm_notebook as tqdm
import tables
from src.network import extract_features
import tensorflow as tf
from src.autoencoder import Autoencoder, EYE_SIZE

PATH_DATA_PREPARED = join(PATH_DATA, 'prepared')


def downscale256to128(image):
    img = image[:, ::2, ::2]
    return img


def get_reference(filename, path_aligned, person_to_files, reference_by_path, sess, autoencoder):
    person = filename.rsplit('-', 1)[0]
    person_files = person_to_files[person]
    if len(person_files) == 1:
        return None, None
    j = 0
    while person_files[j] == filename:
        j += 1
    reference_file = person_files[j]
    path = join(path_aligned, reference_file)
    y = read_image(path)
    y = downscale256to128(y)
    y = np.transpose(y, (1, 2, 0))
    raw_rects = get_rects(reference_by_path[reference_file])
    if raw_rects is None:
        return None, None
    reference_points = np.array(raw_rects).flatten() // 2
    reference_features = sess.run(extract_features(y, reference_points, autoencoder))
    return reference_features, y


def prepare_dataset(start, total_size, reference_by_path, path_aligned, person_to_files, dir, prefix, save_reference_images=False, calc_reference=True):
    sess = tf.Session()
    datasetFile = tables.open_file(join(dir, prefix + '-dataset.h5'), mode='w')
    imagesArray = datasetFile.create_earray(datasetFile.root, 'images', tables.Float32Atom(), (0, 128, 128, 3))
    masksArray = datasetFile.create_earray(datasetFile.root, 'masks', tables.UInt8Atom(), (0, 128, 128, 1))
    pointsArray = datasetFile.create_earray(datasetFile.root, 'points', tables.Int8Atom(), (0, 8))
    referencesArray = datasetFile.create_earray(datasetFile.root, 'references', tables.Float32Atom(), (0, 256))
    referenceImagesArray = datasetFile.create_earray(datasetFile.root, 'reference_images', tables.Float32Atom(), (0, 128, 128, 3))

    is_training = tf.placeholder(tf.bool, [])
    is_left_eye = tf.placeholder(tf.bool, [])
    x_autoencoder = tf.placeholder(tf.float32, [1, EYE_SIZE, EYE_SIZE, 3])
    autoencoder = Autoencoder(x_autoencoder, is_left_eye, is_training, 1)

    images = []
    masks = []
    points = []
    references = []
    filenames = list(reference_by_path.keys())
    i = 0
    for filename in tqdm(filenames[start:start + total_size]):
        path = join(path_aligned, filename)
        image = read_image(path)
        masked = np.zeros((1, 256, 256), dtype=np.uint8)
        make_input_image(masked, reference_by_path[filename], 1)
        rects = get_rects(reference_by_path[filename])
        reference, reference_image = None, None
        if calc_reference:
            reference, reference_image = get_reference(filename, path_aligned, person_to_files, reference_by_path, sess)
        if rects is None or calc_reference and reference is None:
            continue
        lpoints = np.array(rects).flatten()
        image = downscale256to128(image)
        masked = downscale256to128(masked)
        image = np.transpose(image, (1, 2, 0))
        masked = np.transpose(masked, (1, 2, 0))
        imagesArray.append(np.expand_dims(image, 0))
        masksArray.append(np.expand_dims(masked, 0))
        pointsArray.append(np.expand_dims(lpoints.copy() // 2, 0))
        if calc_reference:
            reference = np.array(reference)
            referencesArray.append(np.expand_dims(reference, 0))
        if save_reference_images:
            referenceImagesArray.append(np.expand_dims(reference_image, 0))
    datasetFile.close()


def get_batch_generator(filename, load_reference_images=False, load_references=True):
    def batch_generator(batch_size):
        datasetFile = tables.open_file(filename, mode='r')
        imagesNode = datasetFile.get_node('/images')
        masksNode = datasetFile.get_node('/masks')
        pointsNode = datasetFile.get_node('/points')
        referenceNode = datasetFile.get_node('/references')
        referenceImagesNode = None
        if load_reference_images:
            referenceImagesNode = datasetFile.get_node('/reference_images')
        images = []
        masks = []
        points = []
        references = []
        reference_images = []
        if not load_references:
            for image, mask, point in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows()):
                images.append(image)
                masks.append(mask)
                points.append(point)
                if len(images) == batch_size:
                    yield np.array(images), np.array(masks), np.array(points)
                    images, masks, points, references, reference_images = [], [], [], [], []
        elif load_reference_images:
            for image, mask, point, reference, referenceImage in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows(), referenceNode.iterrows(), referenceImagesNode.iterrows()):
                images.append(image)
                masks.append(mask)
                points.append(point)
                references.append(reference)
                reference_images.append(referenceImage)
                if len(images) == batch_size:
                    yield np.array(images), np.array(masks), np.array(points), np.array(references), np.array(reference_images)
                    images, masks, points, references, reference_images = [], [], [], [], []
        else:
            for image, mask, point, reference in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows(), referenceNode.iterrows()):
                images.append(image)
                masks.append(mask)
                points.append(point)
                references.append(reference)
                if len(images) == batch_size:
                    yield np.array(images), np.array(masks), np.array(points), np.array(references)
                    images, masks, points, references = [], [], [], []
    return batch_generator


def get_full_dataset(path_aligned):
    train_generator = get_batch_generator(join(PATH_DATA_PREPARED, 'train-dataset.h5'))
    test_generator = get_batch_generator(join(PATH_DATA_PREPARED, 'test-dataset.h5'))
    return train_generator, test_generator

def get_initial_train_dataset():
    return get_batch_generator(join(PATH_DATA_PREPARED, 'train-dataset.h5'), False, False) 

def get_final_test_dataset():
    return get_batch_generator(join(PATH_DATA_PREPARED, 'test-dataset.h5'), True)

def prepare_non_reference_train_dataset(path_aligned):
    reference = json.loads(open(join(path_aligned, 'data.json'), 'r').read())
    person_to_files = {}
    for person in reference.keys():
        person_to_files[person] = []
        for image_reference in reference[person]:
            person_to_files[person].append(image_reference['filename'])
    # key - file name, value - map with eyes description
    reference_by_path = dict()
    for person in reference.keys():
        for image_reference in reference[person]:
            reference_by_path[image_reference['filename']] = image_reference
    filenames = list(reference_by_path.keys())
    train_size = len(filenames)
    prepare_dataset(0, train_size, reference_by_path, path_aligned, person_to_files, PATH_DATA_PREPARED, 'train', False, False)

def prepare_full_dataset(path_aligned, train_ratio):
    reference = json.loads(open(join(path_aligned, 'data.json'), 'r').read())
    person_to_files = {}
    for person in reference.keys():
        person_to_files[person] = []
        for image_reference in reference[person]:
            person_to_files[person].append(image_reference['filename'])
    # key - file name, value - map with eyes description
    reference_by_path = dict()
    for person in reference.keys():
        for image_reference in reference[person]:
            reference_by_path[image_reference['filename']] = image_reference
    filenames = list(reference_by_path.keys())
    train_size = int(train_ratio * len(filenames))
    prepare_dataset(0, train_size, reference_by_path, path_aligned, person_to_files, PATH_DATA_PREPARED, 'train')
    prepare_dataset(train_size, len(filenames) - train_size, reference_by_path, path_aligned, person_to_files,
                    PATH_DATA_PREPARED, 'test', True)
    return train_size
