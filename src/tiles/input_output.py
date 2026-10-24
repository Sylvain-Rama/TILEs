import numpy as np
from PIL import Image

from .types import ImageContext


def load_images(image_path: str, heightmap_path: str | None = None, resize: float | None = None) -> ImageContext:
    image = Image.open(image_path)
    if heightmap_path is not None:
        heightmap = Image.open(heightmap_path)
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
