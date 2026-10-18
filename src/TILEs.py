"""Backward-compatible public API for the TILEs package."""

from dithering import dither
from tiles.legacy import quadtree, recursive_slice, throw_polys, voronoitree

__all__ = [
    "dither",
    "quadtree",
    "recursive_slice",
    "throw_polys",
    "voronoitree",
]
