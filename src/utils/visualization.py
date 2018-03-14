import numpy as np
from matplotlib import pyplot as plt
from src.utils.image import convert_to_opencv_format


def show_image(image: np.ndarray, show_immediately: bool = True, title: str = None) -> None:
    """Show image.

    :param image: image to show
    :param show_immediately: whether to force the immediate appearance of the image or not
    :param title: title to show above the image

    """
    image = convert_to_opencv_format(image, convert_color=False)
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    if show_immediately:
        plt.show()
