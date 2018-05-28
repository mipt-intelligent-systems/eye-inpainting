import os
import numpy as np
from src.utils.paths import PROCESSED_NPY_DATA
from src.utils.io import read_image, draw_rectangle, make_input_image, get_rects
import json
from os.path import join
from tqdm import tqdm_notebook as tqdm
import tables

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

def prepare_dataset(start, total_size, reference_by_path, path_aligned, person_to_files, dir, prefix):
    datasetFile = tables.open_file(join(dir, prefix + '-dataset.h5'), mode='w')
    imagesArray = datasetFile.create_earray(datasetFile.root, 'images', tables.Float32Atom(), (0, 128, 128, 3))
    masksArray = datasetFile.create_earray(datasetFile.root, 'masks', tables.UInt8Atom(), (0, 128, 128, 1))
    pointsArray = datasetFile.create_earray(datasetFile.root, 'points', tables.Int8Atom(), (0, 8))
    #referencesArray = datasetFile.create_earray(datasetFile.root, 'references', tables.Float32Atom(), (0, 128, 128, 3))
    images = []
    masks = []
    points = []
    references = []
    filenames = list(reference_by_path.keys())
    for filename in tqdm(filenames[start:start + total_size]):
        path = join(path_aligned, filename)
        image = read_image(path)
        masked = np.zeros((1, 256, 256), dtype=np.uint8)
        make_input_image(masked, reference_by_path[filename], 1)
        rects = get_rects(reference_by_path[filename])
        #reference = get_reference(filename, path_aligned, person_to_files)
        if rects is None: #or reference is None:
            continue
        lpoints = np.array(rects).flatten()
        image = downscale256to128(image)
        masked = downscale256to128(masked)
        image = np.transpose(image, (1, 2, 0))
        masked = np.transpose(masked, (1, 2, 0))
        imagesArray.append(np.expand_dims(image, 0))
        masksArray.append(np.expand_dims(masked, 0))
        pointsArray.append(np.expand_dims(lpoints.copy() // 2, 0))
    datasetFile.close()

def get_batch_generator(filename):
    def batch_generator(batch_size):
        datasetFile = tables.open_file(filename, mode='r')
        imagesNode = datasetFile.get_node('/images')
        masksNode = datasetFile.get_node('/masks')
        pointsNode = datasetFile.get_node('/points')
        images = []
        masks = []
        points = []
        for image, mask, point in zip(imagesNode.iterrows(), masksNode.iterrows(), pointsNode.iterrows()):
            images.append(image)
            masks.append(mask)
            points.append(point)
            if len(images) == batch_size:
                images = np.array(images)
                masks = np.array(masks)
                points = np.array(points)
                yield images, masks, points
                images = []
                masks = []
                points = []
    return batch_generator

def get_full_dataset(path_aligned):
    train_generator = get_batch_generator(join('./data/prepared', 'train-dataset.h5'))
    test_generator = get_batch_generator(join('./data/prepared', 'test-dataset.h5'))
    return train_generator, test_generator

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
    prepare_dataset(0, train_size, reference_by_path, path_aligned, person_to_files, './data/prepared', 'train')
    prepare_dataset(train_size, len(filenames) - train_size, reference_by_path, path_aligned, person_to_files, './data/prepared', 'test')
    return train_size