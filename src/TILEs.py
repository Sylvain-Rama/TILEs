"""Backward-compatible public API for the TILEs package."""

from dithering import dither
from tiles import MeanRGBColorPolicy, MedianRGBColorPolicy, RepresentativeColorPolicy
from tiles.legacy import quadtree, recursive_slice, throw_polys, voronoitree

__all__ = [
    "MeanRGBColorPolicy",
    "MedianRGBColorPolicy",
    "RepresentativeColorPolicy",
    "dither",
    "quadtree",
    "recursive_slice",
    "throw_polys",
    "voronoitree",
]
