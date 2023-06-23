from pathlib import Path

import click
import config
import mosaic
import tile


@click.command()
@click.option(
    "--tilesource",
    type=click.Path(exists=True),
    help="Directory of tile images.",
)
@click.option(
    "--references",
    type=click.Path(exists=True),
    help="Directory of reference images for mosaic creation.",
)
@click.option(
    "--tiles",
    default="",
    help="Directory of cropped images used by the program.",
)
@click.option(
    "--output",
    default="",
    help="Directory of output images.",
)
def main(tilesource: str, references: str, tiles: str, output: str):
    """Simple program to create a mosaic image from a directory of images."""
    tilesource_directory = Path(tilesource)
    references_directory = Path(references)
    tiles_directory = Path(tiles)
    output_directory = Path(output)

    # Directories created by the program if none a specified.
    if not tiles:
        tiles_directory = references_directory.parent / config.DEFAULT_TILES_DIRECTORY
        mosaic.create_necessary_directory(tiles_directory)
    if not output:
        output_directory = references_directory.parent / config.DEFAULT_OUTPUT_DIRECTORY
        mosaic.create_necessary_directory(output_directory)

    tile.prepare_tiles(tilesource=tilesource_directory, tiles_directory=tiles_directory)
    glob = mosaic.glob_directory_images(references_directory)
    mosaic.create_and_write_mosaics(tiles_directory, output_directory, glob)


if __name__ == "__main__":
    main()
