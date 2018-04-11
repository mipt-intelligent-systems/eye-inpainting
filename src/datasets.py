import os
import numpy as np
from src.utils.paths import PROCESSED_NPY_DATA
from src.utils.io import read_image, draw_rectangle, make_input_image, get_rects
import json
from os.path import join
from tqdm import tqdm

def downscale256to128(image):
    img = image[:, ::2, ::2]
    return img

def prepare_one_dataset(start, size, name, reference_by_path, path_aligned):
    images = []
    masks = []
    points = []
    filenames = list(reference_by_path.keys())
    for i, filename in tqdm(enumerate(filenames[start:start + size])):
        path = join(path_aligned, filename)
        y = read_image(path)
        x = np.zeros((1, 256, 256), dtype=np.uint8)
        make_input_image(x, reference_by_path[filename], 1)
        rects = get_rects(reference_by_path[filename])
        if rects is None:
            continue
        lpoints = np.array(rects).flatten()
        y = downscale256to128(y)
        x = downscale256to128(x)
        images.append(y.copy())
        masks.append(x.copy())
        points.append(lpoints.copy() // 2)
    images = np.array(images)
    masks = np.array(masks)
    points = np.array(points)
    np.save(join(PROCESSED_NPY_DATA, 'masks_%s.npy' % name ), masks)
    np.save(join(PROCESSED_NPY_DATA, 'points_%s.npy' % name ), points)
    np.save(join(PROCESSED_NPY_DATA, 'x_%s.npy') % name, images)

def prepare_full_dataset(path_aligned, train_size, test_size):
    reference = json.loads(open(join(path_aligned, 'data.json'), 'r').read())
    # key - file name, value - map with eyes description
    reference_by_path = dict()
    for person in reference.keys():
        for image_reference in reference[person]:
            reference_by_path[image_reference['filename']] = image_reference
    filenames = list(reference_by_path.keys())
    prepare_one_dataset(0, train_size, 'train', reference_by_path, path_aligned)
    prepare_one_dataset(train_size, test_size, 'test', reference_by_path, path_aligned)

def load():
    x_train = np.load(os.path.join(PROCESSED_NPY_DATA, 'x_train.npy'))
    masks_train = np.load(os.path.join(PROCESSED_NPY_DATA, 'masks_train.npy'))
    points_train = np.load(os.path.join(PROCESSED_NPY_DATA, 'points_train.npy'))
    x_test = np.load(os.path.join(PROCESSED_NPY_DATA, 'x_test.npy'))
    masks_test = np.load(os.path.join(PROCESSED_NPY_DATA, 'masks_test.npy'))
    points_test = np.load(os.path.join(PROCESSED_NPY_DATA, 'points_train.npy'))
    return x_train, masks_train, points_train, x_test, masks_test, points_test


if __name__ == '__main__':
    x_train, masks_train, x_test, masks_test = load()
    print(x_train.shape)
    print(x_test.shape)

