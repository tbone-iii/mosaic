""" Contains the core processes behind generating the image tiles to be used
in the mosaic. The primary controlling factors are the pixel resolution
and the image directory. """

import logging
from pathlib import Path
from typing import Generator, Iterable, Tuple

import coloredlogs
import cv2
import numpy as np

from config import PIXEL_RES
from utils import (is_even, iterate_all_image_paths, load_images_from_paths,
                   save_images_with_name)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


def prepare_images(image_dir: Path, cropped_image_dir: Path) -> None:
    """ Part of the main function that prepares any unprepared images. """
    # ? Prepare all images to be square and resized appropriately.
    # Load all images as PIL images.
    image_paths = iterate_all_image_paths(directory=image_dir)
    images = load_images_from_paths(image_paths)

    # Crop all images into a square.
    cropped_images = crop_images_to_square(images=images)

    # Resize each cropped image into the specified dimensions.
    cropped_images = (cv2.resize(image, (PIXEL_RES, PIXEL_RES))
                      for image in cropped_images)

    # Save each cropped image to the prepared images directory with the original name.
    save_images_with_name(images=cropped_images,
                          paths=image_paths,
                          directory=cropped_image_dir)


def crop_images_to_square(images: Iterable[np.ndarray]) -> Generator[np.ndarray, None, None]:
    """ Given an iterable of images, crop each image to a square. """
    count = 0
    for image in images:
        count += 1
        cropped_image = crop_to_square(image)
        logger.debug(f"Cropped image {count} to {cropped_image.shape}.")
        yield cropped_image


def crop_to_square(image: np.ndarray) -> np.ndarray:
    """ Crops cv2 (numpy) images to a square, chopping off edges larger
     than the minimum.
     """
    shape: Tuple[int, int, int] = image.shape
    border = _determine_square_crop_border_indices(shape=shape)
    cropped_image = image[
        border["top"]:border["bottom"],
        border["left"]:border["right"]
    ]

    return cropped_image


def _determine_square_crop_border_indices(shape: Tuple[int, int, int]) -> dict:
    """ Given a tuple of the (x, y) dimensions of an image, determine the crop
        indices for the numpy array as a dict. """
    min_length = min(shape)
    height, width, _ = shape
    chopoff_count = int(abs(height - width) / 2)
    indices = {
        "left": 0,
        "right": width,
        "top": 0,
        "bottom": height,
    }
    if width == min_length:
        # ? Chop off the height
        indices["left"] = 0
        indices["right"] = width
        indices["top"] = chopoff_count
        indices["bottom"] = height - chopoff_count
        if not is_even(height - width):
            indices["bottom"] += 1
    else:
        indices["left"] = chopoff_count
        indices["right"] = width - chopoff_count
        indices["top"] = 0
        indices["bottom"] = height
        if not is_even(height - width):
            indices["right"] += 1
    return indices
