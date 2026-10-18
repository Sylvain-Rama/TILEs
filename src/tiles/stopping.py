from dataclasses import dataclass

from .types import Region, RegionStats


@dataclass(frozen=True)
class ThresholdPolicy:
    standard_deviation: float = 40.0
    max_level: int = 6
    heightmap_threshold: float = 0.1
    heightmap_decay: float = 0.1

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        return (
            stats.standard_deviation < self.standard_deviation
            or region.level >= self.max_level
            or stats.heightmap_max - region.level * self.heightmap_decay < self.heightmap_threshold
        )
