from typing import Protocol

import numpy as np

from .types import ImageContext, Region


class Partitioner(Protocol):
    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]: ...


class StopPolicy(Protocol):
    def should_stop(self, region: Region, stats) -> bool: ...
