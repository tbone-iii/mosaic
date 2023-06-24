import logging
from copy import copy
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Union

import coloredlogs
import cv2
import numpy as np
from config import LOGGING_MODE, VALID_FORMATS

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOGGING_MODE, logger=logger)


def iterate_all_image_paths(directory: Union[Path, str]) -> Iterator:
    """Returns a list of all paths in a directory as pathlib `Path` instances."""
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.exists():
        raise FileExistsError(f"The directory '{directory}' does not exist")

    image_paths: List[Path] = []
    for image_format in VALID_FORMATS:
        pattern = "*" + image_format
        image_paths.extend(directory.glob(pattern))

    logger.info(f"Image paths have been loaded from '{directory.absolute()}'")

    return iter(image_paths)


def load_images_from_paths(paths: Iterable) -> Generator[np.ndarray, None, None]:
    """Loads CV2 images from a list of `pathlib` paths."""
    paths = copy(paths)
    for image_path in paths:
        try:
            image = cv2.imread(str(image_path))
            logger.debug(f"Loaded image: {image_path}")
            yield image
        except OSError:
            logger.warning(f"Image failed to load: {image_path}")
            continue


def save_images_with_name(images: Iterable, paths: Iterable, directory: Path) -> None:
    """Saves each image with the same name as files in the given paths to the
    specified directory.
    """
    path: Path
    image: np.ndarray
    for image, path in zip(images, paths):
        new_path = str(directory / path.name)
        cv2.imwrite(new_path, image)
        logger.debug(f"Saved '{path.name}' with shape {image.shape}")

    logger.info(f"Saved images to '{directory.absolute()}'")


def is_even(value: int) -> bool:
    """Determines whether a given integer is even. Returns a boolean."""
    return bool(value % 2 == 0)
