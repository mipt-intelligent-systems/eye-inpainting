import numpy as np
import cv2
from typing import Tuple


def detect_pixel_interval(image: np.ndarray) -> Tuple[int, int]:
    """Find minimal and maximal pixel value in image array.

    :param image: image
    :return: pair: minimal and maximal pixel values

    """
    return image.min(), image.max()


def convert_pixels_to_ones(image: np.ndarray, interval: Tuple[int, int] = None) -> np.ndarray:
    """Linearly transform image pixel values to fit into interval [-1, 1].

    :param image: image
    :param interval: current interval of pixel values in the image
    :return: image as np.ndarray of floats from interval [-1, 1]

    """
    interval = interval or detect_pixel_interval(image)
    if interval == (-1, 1):
        return image
    center = (interval[0] + interval[1]) / 2
    scale = interval[1] - center
    return np.clip((image - center) / scale, -1, 1)


def convert_pixels_to_zero_one(image: np.ndarray, interval: Tuple[int, int] = None) -> np.ndarray:
    """Linearly transform image pixel values to fit into interval [0, 1].

    :param image: image
    :param interval: current interval of pixel values in the image
    :return: image as np.ndarray of floats from interval [0, 1]

    """
    interval = interval or detect_pixel_interval(image)
    if interval == (0, 1):
        return image
    if interval != (-1, 1):
        image = convert_pixels_to_ones(image, interval=interval)
    return np.clip((image + 1) / 2, 0, 1)


def convert_pixels_to_uint8(image: np.ndarray, interval: Tuple[int, int] = None) -> np.ndarray:
    """Linearly transform image pixel values to fit into interval {0, ..., 255}.

    :param image: image
    :param interval: current interval of pixel values in the image
    :return: image as np.ndarray of np.uint8 from interval [0, 255]

    """
    interval = interval or detect_pixel_interval(image)
    if interval == (0, 255):
        return image
    if interval != (0, 1):
        image = convert_pixels_to_zero_one(image, interval)
    return np.round(image * 255).astype(np.uint8)


def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert image pixels.

    :param image: image as np.ndarray of floats from interval [-1, 1]
    :return: inverted image

    """
    return -image


def move_channels_axis(image: np.ndarray, position: str = 'first') -> np.ndarray:
    """Transpose image axes so that channel axis appears on the specified position.

    :param image: image
    :param position: 'first' or 'last'
    :return: image with channel axis moved

    """
    if position not in {'first', 'last'}:
        raise ValueError('Wrong position. Must be \'first\' or \'last\'')
    permutation = (2, 0, 1) if position == 'first' else (1, 2, 0)
    return np.transpose(image, permutation)


def convert_to_opencv_format(image: np.ndarray, convert_color: bool = True) -> np.ndarray:
    """Convert image to the OpenCV default format.

    The OpenCV format includes:
        * channel axis is the last one;
        * pixels are of type np.uint8;
        * image is in the BGR color map.
    :param image: image in tensor format
    :param convert_color: whether to change the color map to BGR or not
    :return: image in the OpenCV format

    """
    image = move_channels_axis(image, position='last')
    # pixel conversion will implicitly invert the image
    image = invert_image(image)
    image = convert_pixels_to_uint8(image, interval=(-1, 1))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if convert_color else image


def convert_to_tensor_format(image, convert_color=True):
    """Convert image to the tensor format (used for training).

        The tensor format includes:
            * channel axis is the first one;
            * pixels are of type np.float and fit into range [-1, 1];
            * image is in the RGB color map.
        :param image: image in OpenCV format
        :param convert_color: whether to change the color map to RGB or not
        :return: image in the tensor format

        """
    if convert_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = convert_pixels_to_ones(image, interval=(0, 255))
    # pixel conversion implicitly inverts the image
    image = invert_image(image)
    return move_channels_axis(image, position='first')


def get_rect(point, box):
    x1 = point['x'] - box['w'] // 2
    y1 = point['y'] - box['h'] // 2
    x2 = x1 + box['w']
    y2 = y1 + box['h']
    return x1, y1, x2, y2


def draw_rectangle(image, point, box, color):
    x1, y1, x2, y2 = get_rect(point, box)
    image[:, y1:y2, x1:x2] = color
    

def make_input_image(image, reference, color):
    if reference.get('eye_left') != None and reference.get('box_left') != None:
        draw_rectangle(image, reference['eye_left'], reference['box_left'], color)
    if reference.get('eye_right') != None and reference.get('box_right') != None:
        draw_rectangle(image, reference['eye_right'], reference['box_right'], color)


def get_rects(reference):
    if reference.get('eye_left') is None or reference.get('box_left') is None or \
       reference.get('eye_right') is None or reference.get('box_right') is None:
        return None
    else:
        return get_rect(reference['eye_left'], reference['box_left']), get_rect(reference['eye_right'], reference['box_right'])
