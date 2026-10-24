from typing import Protocol

import numpy as np

from .types import ImageContext, Region, RegionStats, RGBColor


class Partitioner(Protocol):
    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]: ...


class StopPolicy(Protocol):
    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator) -> bool: ...


class RepresentativeColorPolicy(Protocol):
    def representative_color(self, rgb_pixels: np.ndarray) -> RGBColor: ...
