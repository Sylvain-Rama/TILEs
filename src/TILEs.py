"""Backward-compatible public API for the TILEs package."""

from dithering import dither
from tiles import (
    DominantColorPolicy,
    MeanRGBColorPolicy,
    MedianRGBColorPolicy,
    MostSaturatedColorPolicy,
    OKLabChromaColorPolicy,
    RepresentativeColorPolicy,
)
from tiles.legacy import quadtree, recursive_slice, throw_polys, voronoitree

__all__ = [
    "DominantColorPolicy",
    "MeanRGBColorPolicy",
    "MedianRGBColorPolicy",
    "MostSaturatedColorPolicy",
    "OKLabChromaColorPolicy",
    "RepresentativeColorPolicy",
    "dither",
    "quadtree",
    "recursive_slice",
    "throw_polys",
    "voronoitree",
]
