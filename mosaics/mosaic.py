import logging
import os
import time
from collections import Counter
from datetime import datetime
from math import inf
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import coloredlogs
import config
import cv2
import numpy as np
import tile
from utils import iterate_all_image_paths

logger = logging.getLogger(__name__)
coloredlogs.install(level=config.LOGGING_MODE, logger=logger)

Vector = Union[Tuple, List]


def color_clusters(image: np.ndarray, K: int = 5) -> Tuple[np.uint8, np.ndarray]:
    """Determines what the average colors are in the image by cluster.

    Returns a tuple of the color clusters and the labels.

    The labels indicate how many of each color "center" are located at particular
    positions. That is, if you add up all the corresponding labels 1 through K, you
    will find that the sum of the quantities of all the same values will give how many
    of each center there is.
    """
    # convert to np.float32
    image_array = np.float32(image)
    temp_image = image_array.reshape((-1, 3))

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        temp_image,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    return centers, labels


def find_dominant_color_from_image(image: np.ndarray) -> tuple:
    """Finds the dominant color of an image given a numpy array of BGR colors.
    Returns a tuple of (B, G, R) color.
    """
    centers, labels = color_clusters(image=image, K=config.CLUSTER_QUANTITY)
    counts: Counter = Counter(labels.flatten())
    # BLUE, GREEN, RED
    dominant_color = tuple(centers[max(counts)])  # type: ignore
    return dominant_color


def find_dominant_color_from_path(image_path: Path) -> tuple:
    """Finds the dominant color of an image given an image path."""
    path = str(image_path)
    image: np.ndarray = cv2.imread(path)
    return find_dominant_color_from_image(image)


def norm(vector: Vector) -> Union[float, int]:
    """Calculates the norm of the given vector."""
    return sum(map(lambda x: x**2, vector)) ** 0.5


def vector_diff(a: Vector, b: Vector) -> Vector:
    """Calculates vector a minus vector b. Returns a vector."""
    return tuple(int(x) - int(y) for x, y in zip(a, b))


def image_to_dominant_colors_from_paths(paths: Iterable[Path]) -> Mapping[Path, Vector]:
    """Given a path of images, find the dominant color of each image and map
    each image path to the dominant color in BGR format.
    """
    dominant_colors: Dict[Path, tuple] = {}
    for image_path in paths:
        dominant_color: tuple = find_dominant_color_from_path(image_path)
        dominant_colors[image_path] = dominant_color
        logger.debug(
            f"Added {dominant_color=} to dominant colors in image '{image_path.name}'"
        )
    return dominant_colors


def prepare_source_image(path: Path) -> np.ndarray:
    """Prepares the target image by cropping it into a square and resizing it into
    the nearest (floor) multiple of the sizes of the small thumbnail images.
    """
    image = cv2.imread(str(path))
    image = tile.crop_to_square(image)
    logger.debug("Cropped target image to a square")

    image = cv2.resize(image, (config.MOSAIC_RESOLUTION, config.MOSAIC_RESOLUTION))
    logger.debug(
        f"Resized image to square of {config.MOSAIC_RESOLUTION=}, "
        f"a multiple of {config.TILE_RESOLUTION=}"
    )
    return image


def get_closest_image_path(dominant_color: tuple, dominant_colors) -> Path:
    """Given the dominant color of an image, find the closest color in the
    collection of dominant colors.

    Returns the path to the image as a `Path` type from the `pathlib` module.
    """
    min_diff = inf
    closest_image_path: Path = Path()
    for path, new_dominant_color in dominant_colors.items():
        color_diff = vector_diff(dominant_color, new_dominant_color)
        diff = norm(color_diff)
        if diff < min_diff:
            min_diff = diff
            closest_image_path = path
    return closest_image_path


def create_mosaic(cropped_image_dir: Path, source_image_path: Path):
    # Get the dominant colors mapping for each immage
    prepared_image_paths = iterate_all_image_paths(cropped_image_dir)
    image_to_dominant_color = image_to_dominant_colors_from_paths(prepared_image_paths)

    # Grab the first SOURCE image in the given image directory.
    source_image = prepare_source_image(path=source_image_path)

    # Going over chunks of an image (a specified grid), average out the color there
    # Determine pixel iteration and boundary.
    tile_resolution = config.TILE_RESOLUTION
    n = source_image.shape[0] // tile_resolution
    pixels = range(0, n * tile_resolution, tile_resolution)

    new_image = source_image.copy()

    t1 = time.perf_counter()

    # ? Determine slices
    slices = []
    pixel_x = 0
    pixel_y = 0
    for pixel_y in pixels:
        for pixel_x in pixels:
            sliced = np.s_[
                pixel_y : pixel_y + tile_resolution, pixel_x : pixel_x + tile_resolution
            ]
            slices.append(sliced)

    for sliced in slices:
        # make an image out of the first square
        sub_image = source_image[sliced]

        dominant_sub_color = find_dominant_color_from_image(image=sub_image)
        closest_image_path = get_closest_image_path(
            dominant_sub_color, image_to_dominant_color
        )

        # Place the image at the location, resized appropriately to some grid.
        new_sub_image = cv2.imread(str(closest_image_path))
        new_image[sliced] = new_sub_image
        logger.debug(f"Added subimage '{closest_image_path}' at ({pixel_x}, {pixel_y})")

    delta_time = time.perf_counter() - t1

    logger.info(f"'{source_image_path.name}' mosaic complete after {delta_time:.2f} secs")
    return new_image


def shrink_image_file_size(image: np.ndarray, output_file_path: str) -> None:
    """Shrinks the file size of the image by reducing the JPEG quality.

    Args:
        image (np.ndarray): Image to shrink the file size of.
        output_file_path (str): Path to the output file.
    """
    jpeg_quality = config.BASE_JPEG_QUALITY
    file_size = os.path.getsize(output_file_path)
    while file_size > config.MAXIMUM_FILE_SIZE:
        jpeg_quality -= 20
        jpeg_quality = max(jpeg_quality, config.MINIMUM_JPEG_QUALITY)
        logger.info(f"Reducing JPEG quality to {jpeg_quality}%")

        cv2.imwrite(
            output_file_path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        if jpeg_quality == config.MINIMUM_JPEG_QUALITY:
            return

        file_size = os.path.getsize(output_file_path)


def create_and_write_mosaics(
    tiles_directory: Path, output_directory: Path, glob: list[Path]
):
    for image_path in glob:
        mosaic_image = create_mosaic(
            cropped_image_dir=tiles_directory,
            source_image_path=image_path,
        )

        # Save file
        todays_datetime = datetime.today().strftime("%Y%m%dT%H%M%S")
        output_file_path = str(
            output_directory
            / f"{todays_datetime}_{image_path.stem}_{config.MOSAIC_IMAGE_SUFFIX}"
        )

        cv2.imwrite(
            output_file_path,
            mosaic_image,
            [cv2.IMWRITE_JPEG_QUALITY, config.BASE_JPEG_QUALITY],
        )
        shrink_image_file_size(mosaic_image, output_file_path)

        # Logging
        logger.info(f"Saved mosaic image to '{output_file_path}'")


def create_necessary_directory(directory: Path):
    """Creates a directory if it does not exist.

    Args:
        directory (Path): The directory to create.
    """
    if not directory.exists():
        logger.info(f"Creating directory '{directory}'")
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def glob_directory_images(directory: Path) -> list[Path]:
    """Returns a list of all the image paths in a directory.

    Args:
        directory (Path): The directory to glob.

    Returns:
        list[Path]: A list of all the image paths in a directory.
    """
    return (
        list(directory.glob("*.png"))
        + list(directory.glob("*.jpg"))
        + list(directory.glob("*.tif"))
    )
