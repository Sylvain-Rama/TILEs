"""Backward-compatible public API for the TILEs package."""

from dithering import dither
from tiles import (
    DominantColorPolicy,
    MeanRGBColorPolicy,
    MedianRGBColorPolicy,
    OKLabChromaColorPolicy,
    RepresentativeColorPolicy,
)
from tiles.legacy import quadtree, recursive_slice, throw_polys, voronoitree

__all__ = [
    "DominantColorPolicy",
    "MeanRGBColorPolicy",
    "MedianRGBColorPolicy",
    "OKLabChromaColorPolicy",
    "RepresentativeColorPolicy",
    "dither",
    "quadtree",
    "recursive_slice",
    "throw_polys",
    "voronoitree",
]
