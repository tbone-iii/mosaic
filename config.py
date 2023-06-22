# Set resized image pixel resolution (length of sq of small images in mosiac)
PIXEL_RES = 50  # pixels

# Number of dominant colors to find using K-Means clustering
CLUSTER_QUANTITY = 1

# Valid image formats to include in processing
VALID_FORMATS = [
    ".png",
    ".jpg",
    ".tif",
]

MOSAIC_IMAGE_NAME = "mosaic.png"

# Target image scale factor
SCALE_FACTOR: int = 5

LOGGING_MODE = "INFO"
