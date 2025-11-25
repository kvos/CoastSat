import math

import numpy as np


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
