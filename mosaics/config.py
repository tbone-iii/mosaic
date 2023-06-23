MOSAIC_RESOLUTION = 10000  # pixels
TILE_RESOLUTION = 80  # pixels, must be a factor of MOSAIC_RESOLUTION

NUMBER_OF_IMAGES = (MOSAIC_RESOLUTION // TILE_RESOLUTION) ** 2

# Number of dominant colors to find using K-Means clustering
CLUSTER_QUANTITY = 1

# Valid image formats to include in processing
VALID_FORMATS = [
    ".png",
    ".jpg",
    ".tif",
]

MOSAIC_IMAGE_SUFFIX = "mosaic.jpg"

LOGGING_MODE = "INFO"

DEFAULT_TILES_DIRECTORY = "./tiles"
DEFAULT_OUTPUT_DIRECTORY = "./mosaics"

MAXIMUM_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
BASE_JPEG_QUALITY = 90  # 0-100%
MINIMUM_JPEG_QUALITY = 10  # 0-100%
