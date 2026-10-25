"""Backward-compatible public API for the TILEs package."""

from dithering import dither
from tiles import (
    DominantColor,
    MeanRGBColor,
    MedianRGBColor,
    MostSaturatedColor,
    OKLabChromaColor,
    RepresentativeColorPolicy,
)
from tiles.legacy import quadtree, recursive_slice, throw_polys, voronoitree

__all__ = [
    "DominantColor",
    "MeanRGBColor",
    "MedianRGBColor",
    "MostSaturatedColor",
    "OKLabChromaColor",
    "RepresentativeColorPolicy",
    "dither",
    "quadtree",
    "recursive_slice",
    "throw_polys",
    "voronoitree",
]
