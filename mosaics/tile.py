""" Contains the core processes behind generating the image tiles to be used
in the mosaic. The primary controlling factors are the pixel resolution
and the image directory. """

import logging
from pathlib import Path
from typing import Generator, Iterable, Tuple

import coloredlogs
import cv2
import numpy as np
from config import TILE_RESOLUTION
from utils import (
    LOGGING_MODE,
    is_even,
    iterate_all_image_paths,
    load_images_from_paths,
    save_images_with_name,
)

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOGGING_MODE, logger=logger)


def prepare_tiles(tilesource: Path, tiles_directory: Path) -> None:
    """Prepares all images in the tilesource directory to be used in the mosaic.

    Makes them into squares and resize them to the specified dimensions.
    """
    image_paths = iterate_all_image_paths(directory=tilesource)
    images = load_images_from_paths(image_paths)
    cropped_images = crop_images_to_square(images=images)
    cropped_images = (
        cv2.resize(image, (TILE_RESOLUTION, TILE_RESOLUTION)) for image in cropped_images
    )
    save_images_with_name(
        images=cropped_images,
        paths=image_paths,
        directory=tiles_directory,
    )


def crop_images_to_square(
    images: Iterable[np.ndarray],
) -> Generator[np.ndarray, None, None]:
    """Given an iterable of images, crop each image to a square."""
    for count, image in enumerate(images, 1):
        cropped_image = crop_to_square(image)
        logger.debug(f"Cropped image {count} to {cropped_image.shape}.")
        yield cropped_image


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """Given a numpy array representing an image, return the dimensions

    Args:
        image (np.ndarray): The image to get the dimensions of.

    Returns:
        Tuple[int, int]: The dimensions of the image.
    """
    return image.shape[:2]


def crop_to_square(image: np.ndarray) -> np.ndarray:
    """Crops cv2 (numpy) images to a square, chopping off edges larger
    than the minimum.
    """
    shape = get_image_dimensions(image)
    border = get_square_crop_border_indices(shape=shape)
    cropped_image = image[
        border["top"] : border["bottom"],
        border["left"] : border["right"],
    ]

    return cropped_image


def get_square_crop_border_indices(shape: Tuple[int, int]) -> dict:
    """Given a tuple of the (x, y) dimensions of an image, determine the crop
    indices for the numpy array as a dict."""
    min_length = min(shape)
    height, width = shape
    chopoff_count = int(abs(height - width) / 2)
    # floored value taken into consideration
    decrement = 0 if is_even(height - width) else 1

    indices = {}
    if width == min_length:
        indices = {
            "left": 0,
            "right": width,
            "top": chopoff_count,
            "bottom": height - chopoff_count - decrement,
        }
    else:
        indices = {
            "left": chopoff_count,
            "right": width - chopoff_count - decrement,
            "top": 0,
            "bottom": height,
        }
    return indices
