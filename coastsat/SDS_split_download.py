import os
import math
import rasterio

import numpy as np

from rasterio.merge import merge


METERS_PER_DEGREE = 111320


def split_area(polygon, sat='S2'):
    aux_left_bound, aux_top_bound = polygon[0]
    aux_right_bound, aux_bot_bound = polygon[2]

    left_bound = min(aux_left_bound, aux_right_bound)
    right_bound = max(aux_left_bound, aux_right_bound)
    bot_bound = min(aux_top_bound, aux_bot_bound)
    top_bound = max(aux_top_bound, aux_bot_bound)

    h_dist = abs(left_bound - right_bound) * METERS_PER_DEGREE
    v_dist = abs(top_bound - bot_bound) * METERS_PER_DEGREE

    if sat == 'S2' or 'COPERNICUS' in sat:
        h_parts = math.ceil(h_dist / 8000)
        v_parts = math.ceil(v_dist / 8000)
    else:
        h_parts = math.ceil(h_dist / 24000)
        v_parts = math.ceil(v_dist / 24000)

    h_offset = h_dist / h_parts if h_parts > 0 else h_dist
    v_offset = v_dist / v_parts if v_parts > 0 else v_dist

    poly_matrix = [[0] * h_parts for _ in range(v_parts)]
    for h in range(h_parts):
        for v in range(v_parts):
            lb = left_bound + h_offset * h / METERS_PER_DEGREE
            rb = left_bound + h_offset * (h + 1) / METERS_PER_DEGREE
            bb = bot_bound + v_offset * v / METERS_PER_DEGREE
            tb = bot_bound + v_offset * (v + 1) / METERS_PER_DEGREE
            poly = [
                [lb, tb],
                [rb, tb],
                [rb, bb],
                [lb, bb],
                [lb, tb]
            ]
            poly_matrix[v_parts - (v + 1)][h] = poly

    return np.array(poly_matrix) if h_parts > 0 or v_parts > 0 else np.array([polygon])


def merge_rasters(raster_list, output_path, method='first'):
    """
    Merge a list of raster (.tif) files into a single raster.

    Parameters:
    - raster_list: list of file paths to input rasters.
    - output_path: file path for the merged output raster.
    - method: how to solve overlaps ('first', 'last', 'min', 'max', 'mean').
    """
    if len(raster_list) == 1:
        os.rename(raster_list[0], output_path)
        return 0

    src_files_to_mosaic = []
    for fp in raster_list:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Merge rasters
    mosaic, out_transform = merge(src_files_to_mosaic, method=method)

    # Copy metadata of first raster and update
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # Write merged raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close opened datasets
    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster created at {output_path}")
