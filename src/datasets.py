import os
import numpy as np
from src.utils.paths import PROCESSED_NPY_DATA, PATH_DATA
from src.utils.io import read_image, draw_rectangle, make_input_image, get_rects
import json
from os.path import join
from tqdm import tqdm_notebook as tqdm
import tables
import tensorflow as tf
from src.autoencoder import Autoencoder, EYE_SIZE

PATH_DATA_PREPARED = join(PATH_DATA, 'prepared')

IMAGE_SIZE = 128
LOCAL_SIZE = 32

def downscale256to128(image):
    img = image[:, ::2, ::2]
    return img

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


def extract_eyes(image, points):
    fx1, fy1, fx2, fy2 = point_to_coords(points.reshape(2, 4)[0])
    sx1, sy1, sx2, sy2 = point_to_coords(points.reshape(2, 4)[1])
    return image[fy1:fy1 + LOCAL_SIZE, fx1:fx1 + LOCAL_SIZE, :], image[sy1:sy1 + LOCAL_SIZE, sx1:sx1 + LOCAL_SIZE, :]

def get_reference(filename, path_aligned, person_to_files, reference_by_path):
    person = filename.rsplit('-', 1)[0]
    person_files = person_to_files[person]
    if len(person_files) == 1:
        return None, None, None
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
        return None, None, None
    reference_points = np.array(raw_rects).flatten() // 2
    left, right = extract_eyes(y, reference_points)
    return left, right, y


def prepare_dataset(start, total_size, reference_by_path, path_aligned, person_to_files, dir, prefix, save_reference_images=False, calc_reference=True):
        
    datasetFile = tables.open_file(join(dir, prefix + '-dataset.h5'), mode='w')
    imagesArray = datasetFile.create_earray(datasetFile.root, 'images', tables.Float32Atom(), (0, 128, 128, 3))
    masksArray = datasetFile.create_earray(datasetFile.root, 'masks', tables.UInt8Atom(), (0, 128, 128, 1))
    pointsArray = datasetFile.create_earray(datasetFile.root, 'points', tables.Int8Atom(), (0, 8))
    referencesArrayLeft = datasetFile.create_earray(datasetFile.root, 'references_left', tables.Float32Atom(), (0, 32, 32, 3))
    referencesArrayRight = datasetFile.create_earray(datasetFile.root, 'references_right', tables.Float32Atom(), (0, 32, 32, 3))
    referenceImagesArray = datasetFile.create_earray(datasetFile.root, 'reference_images', tables.Float32Atom(), (0, 128, 128, 3))

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
        left, right, reference_image = None, None, None
        if calc_reference:
            left, right, reference_image = get_reference(filename, path_aligned, person_to_files, reference_by_path)
        if rects is None or calc_reference and left is None:
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
            left = np.array(left)
            right = np.array(right)
            referencesArrayLeft.append(np.expand_dims(left, 0))
            referencesArrayRight.append(np.expand_dims(right, 0))
        if save_reference_images:
            referenceImagesArray.append(np.expand_dims(reference_image, 0))
    datasetFile.close()


def get_batch_generator(filename, load_reference_images=False, load_references=True):
    def batch_generator(batch_size):
        datasetFile = tables.open_file(filename, mode='r')
        imagesNode = datasetFile.get_node('/images')
        masksNode = datasetFile.get_node('/masks')
        pointsNode = datasetFile.get_node('/points')
        referenceLeftNode = datasetFile.get_node('/references_left')
        referenceRightNode = datasetFile.get_node('/references_right')
        referenceImagesNode = None
        if load_reference_images:
            referenceImagesNode = datasetFile.get_node('/reference_images')
        images = []
        masks = []
        points = []
        referencesLeft = []
        referencesRight = []
        reference_images = []
        if not load_references:
            for image, mask, point in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows()):
                images.append(image)
                masks.append(mask)
                points.append(point)
                if len(images) == batch_size:
                    yield np.array(images), np.array(masks), np.array(points)
                    images, masks, points = [], [], []
        elif load_reference_images:
            for image, mask, point, referenceLeft, referenceRight, referenceImage in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows(), referenceLeftNode.iterrows(), referenceRightNode.iterrows(), referenceImagesNode.iterrows()):
                images.append(image)
                masks.append(mask)
                points.append(point)
                referencesLeft.append(referenceLeft)
                referencesRight.append(referenceRight)
                reference_images.append(referenceImage)
                if len(images) == batch_size:
                    yield np.array(images), np.array(masks), np.array(points), np.array(referencesLeft), np.array(referencesRight), np.array(reference_images)
                    images, masks, points, referencesLeft, referencesRight, reference_images = [], [], [], [], [], []
        else:
            for image, mask, point, referenceLeft, referenceRight in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows(), referenceLeftNode.iterrows(), referenceRightNode.iterrows()):
                images.append(image)
                masks.append(mask)
                points.append(point)
                referencesLeft.append(referenceLeft)
                referencesRight.append(referenceRight)
                if len(images) == batch_size:
                    yield np.array(images), np.array(masks), np.array(points), np.array(referencesLeft), np.array(referencesRight),
                    images, masks, points, referencesLeft, referencesRight, reference_images = [], [], [], [], [], []
    return batch_generator


def get_full_dataset(path_aligned):
    train_generator = get_batch_generator(join(PATH_DATA_PREPARED, 'train-dataset.h5'))
    test_generator = get_batch_generator(join(PATH_DATA_PREPARED, 'test-dataset.h5'))
    return train_generator, test_generator

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
