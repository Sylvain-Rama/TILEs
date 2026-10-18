import numpy as np
from PIL import Image

from .types import ImageContext


def prepare_image(
    image: Image.Image,
    heightmap: Image.Image | None = None,
) -> ImageContext:
    if not isinstance(image, Image.Image):
        raise TypeError("Image must be a valid PIL image")

    rgba_image = image.convert("RGBA")
    if heightmap is None:
        heightmap_image = Image.new("L", rgba_image.size, color=255)
    elif isinstance(heightmap, Image.Image):
        heightmap_image = heightmap.convert("L")
        if heightmap_image.size != rgba_image.size:
            heightmap_image = heightmap_image.resize(rgba_image.size)
    else:
        raise TypeError("Heightmap must be a PIL image or None")

    return ImageContext(
        image=rgba_image,
        rgba=np.asarray(rgba_image),
        heightmap=np.asarray(heightmap_image, dtype=np.float64) / 255.0,
    )
