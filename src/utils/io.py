import numpy as np
import cv2
import os
from os.path import join
from typing import Iterable, Generator

from src.utils.image import convert_to_tensor_format, convert_to_opencv_format, draw_rectangle, make_input_image, get_rects


def read_image(path: str) -> np.ndarray:
    """Read image from file.

    :param path: path to the image file
    :return: image as a numpy.ndarray of floats from interval [-1, 1] of shape (3, height, width)

    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Cannot read image: {path}')
    return convert_to_tensor_format(image)


def save_image(image: np.ndarray, path: str) -> None:
    """Save image to file.

    :param: image: image
    :param path: path to the image file

    """
    image = convert_to_opencv_format(image)
    cv2.imwrite(path, image)


def search_for_extensions(path_root: str, extensions: Iterable[str] = None) -> Generator[str, None, None]:
    """Walk the tree of files and yield all the paths having one of the chosen extensions.

    :param path_root: path to the root of the tree
    :param extensions: iterable of the chosen extensions.
        Each extension must start with a dot: '.jpg'.
        If None, paths of all the files in the tree will be yielded.
    :return: generator of paths

    """
    for root, dirs, files in os.walk(path_root):
        for filename in files:
            extension = os.path.splitext(filename)[1]
            if extensions is None or extension.lower() in extensions:
                yield join(root, filename)
