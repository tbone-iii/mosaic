""" Contains the core processes behind generating the image tiles to be used
in the mosaic. The primary controlling factors are the pixel resolution
and the image directory. """

import logging
from copy import copy
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import coloredlogs
from PIL import Image, ImageOps

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
    cropped_images = (image.resize((PIXEL_RES, PIXEL_RES))
                      for image in cropped_images)

    # Save each cropped image to the prepared images directory with the original name.
    save_images_with_name(images=cropped_images,
                          paths=image_paths,
                          directory=cropped_image_dir)


def crop_images_to_square(images: Iterable[Image.Image]) -> Iterator[Image.Image]:
    """ Given an iterable of images, crop each image to a square. """
    cropped_images: List[Image.Image] = []
    for image in copy(images):
        try:
            cropped_image = crop_to_square(image)
            cropped_images.append(cropped_image)
            logger.debug(f"Cropped {image.fp=} to {cropped_image.size}.")
        except Exception as e:
            logger.error(f"Crop images to square error on image.")
    return iter(cropped_images)


def crop_to_square(image: Image.Image) -> Image.Image:
    """ Crops PIL images to a square, chopping off edges larger than the minimum. """
    size: Tuple[int, int] = image.size
    border = _determine_square_crop_border(size=size)
    try:
        cropped_image = ImageOps.crop(image, border)
    except OSError:
        logger.error(f"Broken data stream when reading image file {image.fp}.")
        raise OSError

    return cropped_image


def _determine_square_crop_border(size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """ Given a tuple of the (x, y) dimensions of an image, determine the crop border
       from each edge necessary to properly crop the image into a square. """
    min_length = min(size)
    width, height = size
    chopoff_count = int(abs(height - width) / 2)
    if width == min_length:
        if is_even(height - width):
            border = (0, chopoff_count, 0, chopoff_count)
        else:
            border = (0, chopoff_count, 0, chopoff_count + 1)
    else:
        if is_even(height - width):
            border = (chopoff_count, 0, chopoff_count, 0)
        else:
            border = (chopoff_count, 0, chopoff_count + 1, 0)
    return border
