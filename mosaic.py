import logging
import time
from collections import Counter
from datetime import datetime
from math import inf
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import coloredlogs
import cv2
import numpy as np

from config import (
    CLUSTER_QUANTITY,
    LOGGING_MODE,
    MOSAIC_IMAGE_NAME,
    PIXEL_RES,
    SCALE_FACTOR,
)
from tiles import crop_to_square, prepare_images
from utils import iterate_all_image_paths

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOGGING_MODE, logger=logger)

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
    centers, labels = color_clusters(image=image, K=CLUSTER_QUANTITY)
    counts: Counter = Counter(labels.flatten())
    # BLUE, GREEN, RED
    dominant_color = tuple(centers[max(counts)])
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


def prepare_target_image(path: Path) -> np.ndarray:
    """Prepares the target image by cropping it into a square and resizing it into
    the nearest (floor) multiple of the sizes of the small thumbnail images.
    """
    image = cv2.imread(str(path))
    image = crop_to_square(image)
    logger.debug("Cropped target image to a square.")
    # Resize the image down to the multiple (GCM) of PIXEL_RES
    new_length = PIXEL_RES * (image.shape[0] // PIXEL_RES) * SCALE_FACTOR
    image = cv2.resize(image, (new_length, new_length))
    logger.debug(f"Resized image to square of {new_length=}, a multiple of {PIXEL_RES=}.")
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
    source_image = prepare_target_image(path=source_image_path)

    # Going over chunks of an image (a specified grid), average out the color there
    # Determine pixel iteration and boundary.
    n = source_image.shape[0] // PIXEL_RES
    pixels = range(0, n * PIXEL_RES, PIXEL_RES)

    new_image = source_image.copy()

    t1 = time.perf_counter()

    # ? Determine slices
    slices = []
    pixel_x = 0
    pixel_y = 0
    for pixel_y in pixels:
        for pixel_x in pixels:
            sliced = np.s_[pixel_y : pixel_y + PIXEL_RES, pixel_x : pixel_x + PIXEL_RES]
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
    with open("times.txt", mode="a") as f:
        f.write(str(delta_time) + "\n")

    logger.info(
        f"'{source_image_path.name}' mosaic complete "
        f" after {round(delta_time, 2):.2f} secs!"
    )
    return new_image


def main():
    # Set directories
    image_dir = Path("./images/")
    cropped_image_dir = Path("./cropped/")
    source_image_dir = Path("./referenceimage/")
    output_dir = Path("./output")

    prepare_images(image_dir=image_dir, cropped_image_dir=cropped_image_dir)

    glob = (
        list(source_image_dir.glob("*.png"))
        + list(source_image_dir.glob("*.jpg"))
        + list(source_image_dir.glob("*.tif"))
    )
    for image_path in glob:
        mosaic_image = create_mosaic(
            cropped_image_dir=cropped_image_dir,
            source_image_path=image_path,
        )

        # Save file
        todays_datetime = datetime.today().strftime("%Y%m%dT%H%M%S")
        output_file_path = str(
            output_dir / f"{todays_datetime}_{image_path.stem}_{MOSAIC_IMAGE_NAME}"
        )
        cv2.imwrite(output_file_path, mosaic_image)


if __name__ == "__main__":
    main()
