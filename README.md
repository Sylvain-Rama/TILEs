# TILEs
Set of scripts to decompose an image in multiple tiles like Quadtrees, Voronoitrees, recursive random slicing, etc...

Algorithms can be set via the local standard deviation or a heightmap

![Quadtrees](https://user-images.githubusercontent.com/60986961/205314503-c91bc8b9-0b7d-498e-81fd-dedec574cd8f.png)

![VoronoiTrees](https://user-images.githubusercontent.com/60986961/205314535-90c64df1-9bba-4f16-b11d-1da276c4ba51.png)

![Lenna](https://github.com/Sylvain-Deposit/TILEs/assets/60986961/4f7af9a4-6d10-4778-a8e7-5dbf5df3c335)

## Performance notes

- Quadtree and recursive slicing compute region statistics in constant time from summed-area tables
  (`tiles.integral`). The tables cost about 40 bytes per pixel and are skipped above
  `tiles.integral.INTEGRAL_MAX_PIXELS` (4 million pixels); results are identical either way, only slower.
- Each region's visible pixels and polygon mask are computed once and shared by the stop policy, the colour policy and the tile image.
- Tile images are rendered when accessed, not stored. `Tile.image` re-renders on every access, and the `images` entry
  returned by `voronoitree` is a lazy `Sequence` (supports `len`, indexing and slicing); wrap it in `list(...)` to render everything at once.

See a [Notebook](https://github.com/Sylvain-Deposit/TILEs/blob/main/TILEs%20example.ipynb) for more information.

This was originally written to use with [PILes](https://github.com/Sylvain-Deposit/PILes)
