import io
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageCms

from .types import ImageContext, TileSet

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _to_srgb(image: Image.Image) -> Image.Image:
    """Convert an RGB image with an embedded ICC profile (e.g. ProPhoto, Adobe RGB) to sRGB.

    Pixel values are otherwise read as sRGB, which desaturates wide-gamut images. The profile is
    dropped from the result so viewers do not apply it a second time.
    """
    icc_profile = image.info.get("icc_profile")
    if not icc_profile or image.mode not in ("RGB", "RGBA"):
        return image
    try:
        converted = ImageCms.profileToProfile(
            image,
            ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode=image.mode,
        )
    except (ImageCms.PyCMSError, OSError):
        return image
    converted.info.pop("icc_profile", None)
    return converted


def load_images(image_path: str, heightmap_path: str | None = None, resize: float | None = None) -> ImageContext:
    image = _to_srgb(Image.open(image_path))
    if heightmap_path is not None:
        heightmap = _to_srgb(Image.open(heightmap_path))
        heightmap = heightmap.resize(image.size)
    else:
        # If no heightmap is provided, create a default white heightmap
        heightmap = Image.new("L", image.size, color=255)

    if (resize is not None) and (resize > 0.0):
        new_size = (int(image.width * resize), int(image.height * resize))
        image = image.resize(new_size)
        heightmap = heightmap.resize(new_size)

    rgba_image = image.convert("RGBA")
    heightmap_image = heightmap.convert("L")

    return ImageContext(
        image=rgba_image,
        rgba=np.asarray(rgba_image),
        heightmap=np.asarray(heightmap_image, dtype=np.float64) / 255.0,
        size=rgba_image.size,  # Simple convenience
        image_path=image_path,
        heightmap_path=heightmap_path,
    )


def tileset_to_figure(
    tile_set: TileSet,
    *,
    canvas_size: tuple[float, float] | None = None,
    width: float | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: float = 150,
    background: str = "white",
) -> "Figure":
    """Render tile polygons as a Matplotlib figure.

    ``width`` and ``figsize`` are in inches and control the physical output size. When ``width``
    is supplied, the height is derived from ``canvas_size`` to preserve the source aspect ratio.
    ``dpi`` controls raster output such as PNG; SVG remains vector output. When ``canvas_size``
    is omitted, the canvas is inferred from the largest tile coordinates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if width is not None and width <= 0:
        raise ValueError("width must be positive")
    if width is not None and figsize is not None:
        raise ValueError("width and figsize cannot both be specified")
    if canvas_size is None:
        if tile_set.tiles:
            points = np.concatenate([tile.region.polygon for tile in tile_set], axis=0)
            canvas_size = (float(np.max(points[:, 0])), float(np.max(points[:, 1])))
        else:
            canvas_size = (1.0, 1.0)
    canvas_width, canvas_height = canvas_size
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas_size must contain positive dimensions")
    if width is not None:
        figsize = (width, width * canvas_height / canvas_width)

    figure, axes = plt.subplots(figsize=figsize, dpi=dpi, facecolor=background)
    polygons = [tile.region.polygon for tile in tile_set]
    colors = [np.asarray(tile.color, dtype=float) / 255.0 for tile in tile_set]
    if polygons:
        axes.add_collection(
            PolyCollection(
                polygons,
                facecolors=colors,
                edgecolors="none",
                antialiaseds=False,
            )
        )
    axes.set_xlim(0, canvas_width)
    axes.set_ylim(canvas_height, 0)
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def save_tileset(
    tile_set: TileSet,
    output_path: str | PathLike[str],
    *,
    canvas_size: tuple[float, float] | None = None,
    width: float | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: float = 150,
    format: str | None = None,
    background: str = "white",
) -> None:
    """Render and save a tile set as PNG or SVG, inferred from ``output_path`` by default."""
    path = Path(output_path)
    output_format = format.lower() if format is not None else path.suffix.removeprefix(".").lower()
    if output_format not in {"png", "svg"}:
        raise ValueError("format must be 'png' or 'svg'")

    figure = tileset_to_figure(
        tile_set,
        canvas_size=canvas_size,
        width=width,
        figsize=figsize,
        dpi=dpi,
        background=background,
    )
    try:
        figure.savefig(path, format=output_format, dpi=dpi, facecolor=background)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
