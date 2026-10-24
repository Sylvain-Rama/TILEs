"""Stop policies deciding when the recursive partitioning of a region ends.

Heightmap values are in [0, 1]; a bright (high) area asks for more detail, a dark one for less.
"""

from dataclasses import dataclass

from .protocols import StopPolicy
from .types import Region, RegionStats


# Legacy threshold-based stop policy
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


@dataclass(frozen=True)
class MaxLevelPolicy:
    """Stop at a fixed recursion depth."""

    max_level: int

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        return region.level >= self.max_level


@dataclass(frozen=True)
class UniformColorPolicy:
    """Stop when the colour deviation of a region is at most `standard_deviation`."""

    standard_deviation: float = 40.0

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        return stats.standard_deviation <= self.standard_deviation


@dataclass(frozen=True)
class HeightmapContrastPolicy:
    """Heightmap-modulated colour tolerance.

    A region stops splitting once its colour deviation is at most
    ``standard_deviation * (1 - h) ** gamma``, with ``h`` the mean heightmap value of the region.
    Dark areas tolerate the full deviation, bright areas keep splitting on tiny differences; a bigger
    ``gamma`` confines the extra detail to the brightest areas. Fully transparent regions stop at once.
    """

    standard_deviation: float = 40.0
    max_level: int = 6
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.standard_deviation < 0:
            raise ValueError("standard_deviation must be non-negative")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        if stats.visible_pixel_count == 0 or region.level >= self.max_level:
            return True
        tolerance = self.standard_deviation * (1.0 - min(max(stats.heightmap_mean, 0.0), 1.0)) ** self.gamma
        return stats.standard_deviation <= tolerance


@dataclass(frozen=True)
class TargetTileSizePolicy:
    """Heightmap-driven tile size, independent of the partitioner and the recursion level.

    The target size goes from ``max_size`` pixels where the heightmap is 0 to ``min_size`` where it is 1
    (``gamma`` bends the curve). A region stops once its longest bounding-box side is at most the target.
    Combine with `UniformColorPolicy` through `AnyOf` to also stop on flat regions.
    """

    max_size: float
    min_size: float = 4.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.min_size <= self.max_size:
            raise ValueError("sizes must satisfy 0 < min_size <= max_size")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")

    def target_size(self, heightmap_mean: float) -> float:
        h = min(max(heightmap_mean, 0.0), 1.0)
        return self.min_size + (self.max_size - self.min_size) * (1.0 - h) ** self.gamma

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        if stats.visible_pixel_count == 0:
            return True
        return max(stats.bbox_size) <= self.target_size(stats.heightmap_mean)


class _Combination:
    def __init__(self, *policies: StopPolicy) -> None:
        if not policies:
            raise ValueError("at least one policy is required")
        self.policies = policies

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.policies))})"


class AnyOf(_Combination):
    """Stop as soon as one of the policies says so."""

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        return any(policy.should_stop(region, stats) for policy in self.policies)


class AllOf(_Combination):
    """Stop only when every policy says so."""

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        return all(policy.should_stop(region, stats) for policy in self.policies)
