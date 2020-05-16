from tiles import crop_to_square
import logging
from collections import Counter
from math import inf
from pathlib import Path
from typing import Dict, List, Tuple, Union

import coloredlogs
import cv2
import numpy as np
from PIL import Image

from config import CLUSTER_QUANTITY, PIXEL_RES
from utils import iterate_all_image_paths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


def color_clusters(image: np.ndarray, K: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """ Determines what the average colors are in the image by cluster.

    Returns a tuple of the color clusters and the labels.

    The labels indicate how many of each color "center" are located at particular
    positions. That is, if you add up all the corresponding labels 1 through K, you
    will find that the sum of the quantities of all the same values will give how many
    of each center there is.
    """
    # convert to np.float32
    image = np.float32(image)
    temp_image = image.reshape((-1, 3))

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(
        temp_image,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    return centers, labels


def _find_dominant_color(image: np.ndarray):
    """ Finds the dominant color of an image given a numpy array of BGR colors. """
    centers, labels = color_clusters(image=image, K=CLUSTER_QUANTITY)
    counts: Counter = Counter(labels.flatten())
    # BLUE, GREEN, RED
    dominant_color = tuple(centers[max(counts)])
    # CONVERTS TO RED, GREEN, BLUE
    return dominant_color[::-1]


def find_dominant_color_from_path(image_path: Path) -> tuple:
    """ Finds the dominant color of an image given an image path. """
    path = str(image_path)
    image: np.ndarray = cv2.imread(path)
    return _find_dominant_color(image)


def find_dominant_color_from_image(image: np.ndarray) -> tuple:
    return _find_dominant_color(image)


Vector = Union[Tuple, List]


def norm(vector: Vector) -> Union[float, int]:
    """ Calculates the norm of the given vector. """
    return sum(map(lambda x: x**2, vector))


def vector_diff(a: Vector, b: Vector) -> Vector:
    """ Calculates vector a minus vector b. Returns a vector."""
    return tuple(int(x) - int(y) for x, y in zip(a, b))


def convert_image_PIL2OpenCV(pil_image: Image):
    """ Converts a PIL image to an Open CV image.

    Returns an Open CV image.
    """
    pil_image = pil_image.convert("RGB")
    open_cv_image = np.array(pil_image)
    # Convert "RGB" to "BGR"
    return open_cv_image[:, :, ::-1].copy()


def process_images_for_mosaic(cropped_image_dir: Path, target_image_dir: Path):
    # Grab the path of each modified image
    prepared_image_paths = iterate_all_image_paths(cropped_image_dir)

    # Find the average color value of each image
    # and map the image path to the dominant color
    dominant_colors: Dict[Path, tuple] = {}
    for cropped_image_path in prepared_image_paths:
        dominant_color: tuple = find_dominant_color_from_path(cropped_image_path)
        dominant_colors[cropped_image_path] = dominant_color
        logger.debug(
            f"Added {dominant_color=} to dominant colors "
            f"in image '{cropped_image_path.name}'"
        )

    # Going over chunks of an image (a specified grid), average out the color there
    # Force target image to be a square and with dimensions being a multiple of PIXEL_RES.
    target_image = next(iterate_all_image_paths(target_image_dir))
    image = Image.open(target_image)
    image = crop_to_square(image)
    logger.debug("Cropped target image to a square.")
    image.save("mosaic-square.png")
    # Resize the image down to the multiple of PIXEL_RES
    new_length = PIXEL_RES * (image.size[0] // PIXEL_RES)
    image = image.resize((new_length, new_length))

    logger.debug(f"Resized image to square of {new_length=}, a multiple of {PIXEL_RES=}.")

    # Convert to an Open CV image
    open_cv_image = convert_image_PIL2OpenCV(pil_image=image)
    logger.debug(f"Converted PIL image to OpenCV image with shape {open_cv_image.shape}.")

    # Determine pixel iteration and boundary.
    n = new_length // PIXEL_RES
    pixels = range(0, n * PIXEL_RES, PIXEL_RES)

    # TODO: THis can be vectorized/parllelized
    for pixel_y in pixels:
        for pixel_x in pixels:
            # make an image out of the first square
            sub_image = open_cv_image[
                pixel_y:pixel_y + PIXEL_RES,
                pixel_x:pixel_x + PIXEL_RES,
            ]

            dominant_sub_color = find_dominant_color_from_image(
                image=sub_image
            )

            # Find in the set the closest color value to the averaged color chunk.
            min_diff = inf
            closest_image_path: Path
            for path, dominant_color in dominant_colors.items():
                color_diff = vector_diff(dominant_color, dominant_sub_color)
                diff = norm(color_diff)
                if diff < min_diff:
                    min_diff = diff
                    closest_image_path = path

            # Place the image at the location, resized appropriately to some grid.
            new_sub_image = cv2.imread(str(closest_image_path))
            open_cv_image[pixel_y:pixel_y + PIXEL_RES,
                          pixel_x:pixel_x + PIXEL_RES] = new_sub_image
            logger.debug(
                f"Added subimage '{closest_image_path}' at ({pixel_x}, {pixel_y})")

    logger.info(f"Image finished!")
    return open_cv_image


def main():
    # Set directories
    image_dir = Path("./images/")
    cropped_image_dir = Path("./cropped/")
    target_image_dir = Path("./mosaicimage/")

    # prepare_images(image_dir=image_dir,
    #                cropped_image_dir=cropped_image_dir)

    open_cv_image = process_images_for_mosaic(cropped_image_dir=cropped_image_dir,
                                              target_image_dir=target_image_dir)

    cv2.imwrite("mosaic.png", open_cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
