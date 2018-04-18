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

def get_reference(filename, path_aligned, person_to_files):
    person = filename.rsplit('-', 1)[0]
    person_files = person_to_files[person]
    if len(person_files) == 1:
        return None
    j = 0
    while person_files[j] == filename:
        j += 1
    reference_file = person_files[j]
    path = join(path_aligned, reference_file)
    y = read_image(path)
    y = downscale256to128(y)
    return y

def get_batch_generator(start, total_size, reference_by_path, path_aligned, person_to_files):
    def batch_generator(batch_size):
        images = []
        masks = []
        points = []
        references = []
        filenames = list(reference_by_path.keys())
        for i, filename in enumerate(filenames[start:start + total_size]):
            path = join(path_aligned, filename)
            y = read_image(path)
            x = np.zeros((1, 256, 256), dtype=np.uint8)
            make_input_image(x, reference_by_path[filename], 1)
            rects = get_rects(reference_by_path[filename])
            reference = get_reference(filename, path_aligned, person_to_files)
            if rects is None or reference is None:
                continue
            lpoints = np.array(rects).flatten()
            y = downscale256to128(y)
            x = downscale256to128(x)
            images.append(y)
            masks.append(x)
            references.append(reference)
            points.append(lpoints.copy() // 2)
            if len(images) == batch_size:
                images = np.array(images)
                masks = np.array(masks)
                points = np.array(points)
                references = np.array(references)
                images = np.transpose(images, (0, 2, 3, 1)) 
                masks = np.transpose(masks, (0, 2, 3, 1)) 
                references = np.transpose(references, (0, 2, 3, 1))
                yield images, masks, points, references
                images = []
                masks = []
                points = []
                references = []
    return batch_generator

def get_full_dataset(path_aligned, train_ratio):
    reference = json.loads(open(join(path_aligned, 'data.json'), 'r').read())
    # key - file name, value - map with eyes description
    person_to_files = {}
    for person in reference.keys():
        person_to_files[person] = []
        for image_reference in reference[person]:
            person_to_files[person].append(image_reference['filename'])
    reference_by_path = dict()
    for person in reference.keys():
        for image_reference in reference[person]:
            reference_by_path[image_reference['filename']] = image_reference
    filenames = list(reference_by_path.keys())
    train_size = int(train_ratio * len(filenames))
    train_generator = get_batch_generator(0, train_size, reference_by_path, path_aligned, person_to_files)
    test_generator = get_batch_generator(train_size, len(filenames) - train_size, reference_by_path, path_aligned, person_to_files)
    return train_generator, test_generator, train_size

