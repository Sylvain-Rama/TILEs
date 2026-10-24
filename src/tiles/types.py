from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from PIL import Image

from .integral import IntegralImage, build_integral_image

RGBColor = tuple[int, int, int]


@dataclass(frozen=True)
class ImageContext:
    image: Image.Image
    rgba: np.ndarray
    heightmap: np.ndarray
    size: tuple[int, int]
    image_path: str
    heightmap_path: str | None

    @cached_property
    def integral(self) -> IntegralImage | None:
        """Summed-area tables for constant-time rectangle statistics; None for very large images."""
        return build_integral_image(self.rgba)

    @cached_property
    def visible_heightmap(self) -> np.ndarray:
        """Heightmap with -inf on transparent pixels, so a plain max ignores them."""
        visible = self.rgba[:, :, 3] != 0
        if visible.all():
            return self.heightmap
        return np.where(visible, self.heightmap, -np.inf)

    @cached_property
    def masked_heightmap(self) -> np.ndarray:
        """Heightmap with 0 on transparent pixels, so a plain sum only counts visible ones."""
        visible = self.rgba[:, :, 3] != 0
        if visible.all():
            return self.heightmap
        return np.where(visible, self.heightmap, 0.0)


@dataclass(frozen=True)
class Region:
    polygon: np.ndarray
    level: int = 0
    parent_id: int | None = None


@dataclass(frozen=True)
class RegionStats:
    standard_deviation: float
    mean_color: RGBColor
    heightmap_max: float
    visible_pixel_count: int
    heightmap_mean: float = 0.0
    bbox_size: tuple[int, int] = (0, 0)


@dataclass
class Tile:
    region: Region
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    color: RGBColor
    # Either an image or a callable rendering one; rendering on access keeps tiles light in memory.
    image_source: Image.Image | Callable[[], Image.Image] | None = None

    @property
    def image(self) -> Image.Image | None:
        source = self.image_source
        return source() if callable(source) else source


@dataclass(frozen=True)
class TileSet:
    tiles: tuple[Tile, ...]
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    def as_legacy_dict(self) -> dict[str, list[Any]]:
        from .legacy import tiles_to_legacy_dict

        return tiles_to_legacy_dict(self.tiles)
