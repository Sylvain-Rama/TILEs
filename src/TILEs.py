"""Backward-compatible top-level API for the TILEs package."""

from dithering import dither
from tiles import *
from tiles import __all__ as _tiles_all

__all__ = list(_tiles_all) + ["dither"]
