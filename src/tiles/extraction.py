from PIL import Image, ImageDraw

from .geometry import polygon_bbox, polygon_center, translate_polygon
from .types import ImageContext, Region, Tile


def extract_tile(context: ImageContext, region: Region) -> Tile:
    image_height, image_width = context.rgba.shape[:2]
    bbox = polygon_bbox(region.polygon, image_width, image_height)
    left, top = bbox[:2]
    cropped = context.image.crop(bbox)
    polygon = translate_polygon(region.polygon, left, top)

    mask = Image.new("L", cropped.size, 0)
    ImageDraw.Draw(mask).polygon(
        [tuple(point) for point in polygon],
        fill=255,
    )
    cropped.putalpha(mask)

    return Tile(
        region=region,
        bbox=bbox,
        center=polygon_center(region.polygon),
        color=(0, 0, 0),
        image=cropped,
    )
