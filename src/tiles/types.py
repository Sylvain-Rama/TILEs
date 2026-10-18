from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageContext:
    image: Image.Image
    rgba: np.ndarray
    heightmap: np.ndarray


@dataclass(frozen=True)
class Region:
    polygon: np.ndarray
    level: int = 0
    parent_id: int | None = None


@dataclass(frozen=True)
class RegionStats:
    standard_deviation: float
    mean_color: tuple[int, int, int]
    heightmap_max: float
    visible_pixel_count: int


@dataclass
class Tile:
    region: Region
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    color: tuple[int, int, int]
    image: Image.Image | None = None


@dataclass(frozen=True)
class TileSet:
    tiles: tuple[Tile, ...]

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    def as_legacy_dict(self) -> dict[str, list[Any]]:
        from .legacy import tiles_to_legacy_dict

        return tiles_to_legacy_dict(self.tiles)
