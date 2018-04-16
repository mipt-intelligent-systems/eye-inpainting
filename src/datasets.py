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

def get_batch_generator(start, total_size, reference_by_path, path_aligned):
    def batch_generator(batch_size):
        images = []
        masks = []
        points = []
        filenames = list(reference_by_path.keys())
        for i, filename in enumerate(filenames[start:start + total_size]):
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
            if len(images) == batch_size:
                images = np.array(images)
                masks = np.array(masks)
                points = np.array(points)
                images = np.transpose(images, (0, 2, 3, 1)) 
                masks = np.transpose(masks, (0, 2, 3, 1)) 
                yield images, masks, points
                images = []
                masks = []
                points = []
    return batch_generator

def get_full_dataset(path_aligned, train_ratio):
    reference = json.loads(open(join(path_aligned, 'data.json'), 'r').read())
    # key - file name, value - map with eyes description
    reference_by_path = dict()
    for person in reference.keys():
        for image_reference in reference[person]:
            reference_by_path[image_reference['filename']] = image_reference
    filenames = list(reference_by_path.keys())
    train_size = int(train_ratio * len(filenames))
    train_generator = get_batch_generator(0, train_size, reference_by_path, path_aligned)
    test_generator = get_batch_generator(train_size, len(filenames) - train_size, reference_by_path, path_aligned)
    return train_generator, test_generator, train_size

