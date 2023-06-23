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

MOSAIC_IMAGE_SUFFIX = "mosaic.png"

LOGGING_MODE = "INFO"

DEFAULT_TILES_DIRECTORY = "./tiles"
DEFAULT_OUTPUT_DIRECTORY = "./mosaics"
