"""Stop policies deciding when the recursive partitioning of a region ends.

Heightmap values are in [0, 1]; a bright (high) area asks for more detail, a dark one for less.
"""

import math
from dataclasses import dataclass

import numpy as np

from .protocols import StopPolicy
from .types import Region, RegionStats


# Legacy threshold-based stop policy
@dataclass(frozen=True)
class ThresholdPolicy:
    standard_deviation: float = 40.0
    max_level: int = 6
    heightmap_threshold: float = 0.1
    heightmap_decay: float = 0.1

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        return (
            stats.standard_deviation < self.standard_deviation
            or region.level >= self.max_level
            or stats.heightmap_max - region.level * self.heightmap_decay < self.heightmap_threshold
        )


@dataclass(frozen=True)
class MaxLevelPolicy:
    """Stop at a fixed recursion depth."""

    max_level: int

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        return region.level >= self.max_level


@dataclass(frozen=True)
class UniformColorPolicy:
    """Stop when the colour deviation of a region is at most `standard_deviation`."""

    standard_deviation: float = 40.0

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
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

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
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

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        if stats.visible_pixel_count == 0:
            return True
        return max(stats.bbox_size) <= self.target_size(stats.heightmap_mean)


@dataclass(frozen=True)
class HeightmapRangePolicy:
    """Stop when the heightmap is nearly flat across the region (``max - min <= max_range``).

    Regions straddling a heightmap edge or gradient are not flat, so they keep splitting and the
    edge stays sharp. On its own this ignores colour; ``AllOf(UniformColorPolicy(...), HeightmapRangePolicy(...))``
    keeps splitting while either the colours or the heightmap vary too much.
    """

    max_range: float = 0.1

    def __post_init__(self) -> None:
        if self.max_range < 0:
            raise ValueError("max_range must be non-negative")

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        return stats.visible_pixel_count == 0 or stats.heightmap_range <= self.max_range


@dataclass(frozen=True)
class StochasticStopPolicy:
    """Stop at random, more often in dark areas than in bright ones.

    Past ``min_level``, a region stops with probability
    ``bright_probability + (dark_probability - bright_probability) * (1 - h) ** gamma``, ``h`` being its
    mean heightmap value. There is no hard threshold, so the tile sizes change smoothly instead of
    forming bands. Draws come from the generator `grow_tree` seeds, so a seed reproduces the result.
    This policy is never satisfied by colour alone: combine it with `UniformColorPolicy` through `AnyOf`
    to also stop on flat regions.
    """

    dark_probability: float = 0.8
    bright_probability: float = 0.1
    max_level: int = 6
    min_level: int = 0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.bright_probability <= 1.0 and 0.0 <= self.dark_probability <= 1.0):
            raise ValueError("probabilities must be between 0 and 1")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.min_level < 0:
            raise ValueError("min_level must be non-negative")

    def stop_probability(self, heightmap_mean: float) -> float:
        h = min(max(heightmap_mean, 0.0), 1.0)
        return self.bright_probability + (self.dark_probability - self.bright_probability) * (1.0 - h) ** self.gamma

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        if stats.visible_pixel_count == 0 or region.level >= self.max_level:
            return True
        if region.level < self.min_level:
            return False
        if rng is None:
            raise ValueError("StochasticStopPolicy needs the random generator passed by grow_tree")
        return bool(rng.random() < self.stop_probability(stats.heightmap_mean))


@dataclass(frozen=True)
class PerceptualUniformityPolicy:
    """Stop when the colours of a region are perceptually close to each other.

    The error is the RMS OKLab distance to the region's mean colour,
    ``sqrt(lightness_weight * var(L) + var(a) + var(b))``. OKLab lightness spans 0..1, and a difference
    of about 0.02 is just noticeable. ``lightness_weight`` above 1 makes brightness changes count more
    than hue changes, below 1 less. Unlike the RGB deviation, this does not over-react to saturated blues
    and under-react to greens.
    """

    tolerance: float = 0.05
    lightness_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.lightness_weight < 0:
            raise ValueError("lightness_weight must be non-negative")

    def error(self, stats: RegionStats) -> float:
        lightness, a, b = stats.oklab_variance
        return math.sqrt(self.lightness_weight * lightness + a + b)

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        return stats.visible_pixel_count == 0 or self.error(stats) <= self.tolerance


class _Combination:
    def __init__(self, *policies: StopPolicy) -> None:
        if not policies:
            raise ValueError("at least one policy is required")
        self.policies = policies

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.policies))})"


class AnyOf(_Combination):
    """Stop as soon as one of the policies says so."""

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        return any(policy.should_stop(region, stats, rng) for policy in self.policies)


class AllOf(_Combination):
    """Stop only when every policy says so."""

    def should_stop(self, region: Region, stats: RegionStats, rng: np.random.Generator | None = None) -> bool:
        return all(policy.should_stop(region, stats, rng) for policy in self.policies)
