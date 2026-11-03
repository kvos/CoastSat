#!/usr/bin/env python
"""
Auto-generate shore-normal transects for CoastSat.

Takes your reference shoreline, puts a transect every 100 m and saves them as
a .geojson you can load with SDS_tools.transects_from_geojson. Each transect
starts on land and ends in the sea. The sea side is found from MNDWI on a few
of your downloaded images, or you can set it with --sea-side left/right.

Run from the CoastSat folder:
    python examples/transect_generation/auto_generate_transects.py --sitename NARRA

Author: Abhishek Anand, PhD student, University of California, Irvine (2026)
"""
import os
import sys
import pickle
import argparse
import warnings

import numpy as np
import geopandas as gpd
import pyproj
from shapely.geometry import LineString

# Settings you may want to change
MAX_GAP = 500     # cut the shoreline at jumps longer than this (m), e.g. across a river mouth
SMOOTH = 100      # smooth the shoreline over this distance (m) so transects don't zig-zag
N_IMAGES = 5      # number of clear images used to find the sea side
MAX_CLOUD = 0.1   # maximum cloud cover of those images
SATELLITES = ['S2', 'L8', 'L9', 'L5']


def load_shoreline(shoreline, epsg=None):
    """Read the shoreline and return it as a list of lines (Nx2 arrays) and its EPSG."""
    # CoastSat's .pkl (or settings['reference_shoreline']) has no CRS, so we need the EPSG
    if isinstance(shoreline, str) and shoreline.endswith('.pkl'):
        with open(shoreline, 'rb') as f:
            shoreline = pickle.load(f)
    if not isinstance(shoreline, str):
        if epsg is None:
            raise ValueError('give --epsg when the shoreline is a .pkl file or an array')
        return [np.asarray(shoreline, float)[:, :2]], int(epsg)

    # a line file (.geojson, .shp...), moved to a metric CRS if it is in lon/lat
    gdf = gpd.read_file(shoreline)
    if epsg:
        gdf = gdf.to_crs(epsg=epsg)
    elif gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    lines = gdf.geometry.explode(index_parts=False)
    return [np.array(l.coords)[:, :2] for l in lines if len(l.coords) > 1], gdf.crs.to_epsg()


def split_at_gaps(xy):
    """Cut the line where it jumps, so no transects are placed across a river mouth."""
    # CoastSat joins separately drawn lines with a straight jump. A jump has to be
    # long AND much longer than the usual point spacing, so a straight stretch
    # drawn with a few clicks is not cut.
    step = np.hypot(*np.diff(xy, axis=0).T)
    gaps = (step > MAX_GAP) & (step > 5 * np.median(step))
    return [seg for seg in np.split(xy, np.where(gaps)[0] + 1) if len(seg) > 1]


def smooth(xy):
    """Resample the line every 5 m and take a moving average over SMOOTH metres."""
    line = LineString(xy)
    pts = np.array([line.interpolate(s).coords[0][:2]
                    for s in np.linspace(0, line.length, int(line.length // 5) + 2)])
    k = int(SMOOTH / 5) | 1   # odd window
    if len(pts) <= k:
        return line
    # mirror the ends so they don't move
    pts = np.pad(pts, ((k // 2, k // 2), (0, 0)), mode='reflect', reflect_type='odd')
    w = np.ones(k) / k
    return LineString(np.column_stack([np.convolve(pts[:, 0], w, 'valid'),
                                       np.convolve(pts[:, 1], w, 'valid')]))


def make_transects(segments, spacing):
    """Put a transect every `spacing` m, at right angles to the shoreline.
    For now the sea is assumed to be on the right; orient() fixes it."""
    transects = []
    for seg_id, xy in enumerate(segments):
        line = smooth(xy)
        L = line.length
        if L < spacing:
            continue
        for s in np.arange((L % spacing) / 2, L + 1e-6, spacing):   # centred on the segment
            p = np.array(line.interpolate(s).coords[0])
            a = np.array(line.interpolate(max(s - 25, 0)).coords[0])
            b = np.array(line.interpolate(min(s + 25, L)).coords[0])
            tx, ty = (b - a) / np.hypot(*(b - a))                    # direction along the shore
            transects.append({'segment': seg_id, 'centre': p,
                              'normal': np.array([ty, -tx]),          # 90 degrees to the right
                              'check': False})
    return transects


def clear_images(sitename, filepath):
    """Give the MNDWI of a few clear images that CoastSat has downloaded."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from coastsat import SDS_download, SDS_preprocess, SDS_tools

    inputs = {'sitename': sitename, 'filepath': filepath}
    metadata = SDS_download.get_metadata(inputs)
    images = [(sat, i) for sat in SATELLITES if sat in metadata
              for i in range(len(metadata[sat]['filenames']))]
    found = 0
    # random (but repeatable) order so the images are spread over the years
    for k in np.random.default_rng(0).permutation(len(images))[:50]:
        sat, i = images[k]
        fn = SDS_tools.get_filenames(metadata[sat]['filenames'][i],
                                     SDS_tools.get_filepath(inputs, sat), sat)
        try:
            im_ms, georef, cloud, _, _, nodata = SDS_preprocess.preprocess_single(fn, sat, False, False, 40)
        except Exception:
            continue   # skip images that can't be read
        if np.sum(cloud & ~nodata) > MAX_CLOUD * np.sum(~nodata):
            continue   # too cloudy
        # MNDWI = (green - swir) / (green + swir), positive over water
        yield {'mndwi': SDS_tools.nd_index(im_ms[:, :, 1], im_ms[:, :, 4], cloud),
               'georef': georef, 'epsg': int(metadata[sat]['epsg'][i]),
               'date': metadata[sat]['dates'][i], 'sat': sat}
        found += 1
        if found == N_IMAGES:
            return


def mndwi_at(xy, img, epsg):
    """MNDWI of the pixel under each point (NaN if cloudy or outside the image)."""
    shape = xy.shape[:-1]
    x, y = xy.reshape(-1, 2).T
    if img['epsg'] != epsg:
        x, y = pyproj.Transformer.from_crs(epsg, img['epsg'], always_xy=True).transform(x, y)
    g, im = img['georef'], img['mndwi']   # georef = [x0, pixel width, 0, y0, 0, pixel height]
    col = np.floor((x - g[0]) / g[1]).astype(int)
    row = np.floor((y - g[3]) / g[5]).astype(int)
    inside = (row >= 0) & (row < im.shape[0]) & (col >= 0) & (col < im.shape[1])
    vals = np.full(x.shape, np.nan)
    vals[inside] = im[row[inside], col[inside]]
    return vals.reshape(shape)


def votes_on_image(transects, img, epsg, land_length, sea_length):
    """Each transect votes +1 if its sea half is wetter than its land half,
    -1 if the land half is wetter, 0 if the image can't tell."""
    c = np.array([t['centre'] for t in transects])[:, None]
    n = np.array([t['normal'] for t in transects])[:, None]
    f = np.array([0.25, 0.5, 0.75, 1.0])[None, :, None]   # 4 points along each half
    sea = mndwi_at(c + n * f * sea_length, img, epsg)
    land = mndwi_at(c - n * f * land_length, img, epsg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        diff = np.nanmean(sea, 1) - np.nanmean(land, 1)
    ok = (np.isfinite(sea).sum(1) >= 2) & (np.isfinite(land).sum(1) >= 2)
    return np.where(ok & (diff > 0.1), 1, np.where(ok & (diff < -0.1), -1, 0))


def orient(transects, votes):
    """Flip each segment whose votes say the sea is on the left.
    A drawn line has one direction, so we decide per segment. With many
    transects and several images, a few clouds or a lagoon can't swing it."""
    votes = np.array(votes)   # images x transects
    seg = np.array([t['segment'] for t in transects])
    for s in np.unique(seg):
        v = votes[:, seg == s]
        right, left = np.sum(v > 0), np.sum(v < 0)
        unsure = right + left < 5 or max(right, left) < 0.6 * (right + left)
        flip = not unsure and left > right
        if unsure:
            print(f'  segment {s}: not sure ({right + left} votes), check it in the QC figure')
        else:
            print(f'  segment {s}: sea on the {"left" if flip else "right"}, '
                  f'{max(right, left)}/{right + left} votes agree')
        for i in np.where(seg == s)[0]:
            if flip:
                transects[i]['normal'] = -transects[i]['normal']
            mine = votes[:, i] * (-1 if flip else 1)
            # flag it if we're unsure, or if this transect disagrees with its segment
            transects[i]['check'] = unsure or np.sum(mine < 0) > np.sum(mine > 0)


def plot_qc(transects, ends, segments, fname, output=None):
    """Save the same transect figure as the CoastSat example: mapped shorelines,
    transects and a blue dot at the landward end (origin) of each transect."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[15, 8], tight_layout=True)
    plt.axis('equal')
    plt.xlabel('Eastings')
    plt.ylabel('Northings')
    plt.grid(linestyle=':', color='0.5')
    if output is not None:   # all the shorelines CoastSat mapped
        for sl, date in zip(output['shorelines'], output['dates']):
            if len(sl) > 0:
                plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))
    else:                    # no mapped shorelines yet, show the reference shoreline
        for xy in segments:
            plt.plot(xy[:, 0], xy[:, 1], '-', color='0.5')
    every = max(1, len(transects) // 10)   # label about 10 transects so they stay readable
    pts = np.array([p for t in transects for p in ends(t)])
    min_gap = 0.05 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    last = None
    for j, t in enumerate(transects):
        tr = np.array(ends(t))
        plt.plot(tr[0, 0], tr[0, 1], 'bo', ms=5)
        plt.plot(tr[:, 0], tr[:, 1], '-', color='r' if t['check'] else 'k', lw=1)   # red = check it
        # skip a label if it would sit on top of the previous one
        if j % every == 0 and (last is None or np.hypot(*(tr[0] - last)) > min_gap):
            plt.text(tr[0, 0] - 100, tr[0, 1] + 100, t['name'],
                     va='center', ha='right', bbox=dict(boxstyle="square", ec='k', fc='w'))
            last = tr[0]
    fig.savefig(fname, dpi=200)
    return fig


def auto_generate_transects(shoreline=None, epsg=None, sitename=None, filepath='data',
                            sea_side='auto', spacing=100, land_length=200, sea_length=200,
                            out=None, prefix=None, output=None):
    """Make the transects, save them as .geojson and return them in the same
    format as SDS_tools.transects_from_geojson: {name: [[land x, y], [sea x, y]]}.

    Abhishek Anand, UC Irvine 2026
    """
    if not sitename and (shoreline is None or sea_side == 'auto'):
        raise ValueError('give --sitename, or a --shoreline with --sea-side left/right')
    if shoreline is None:   # the one SDS_preprocess.get_reference_sl saved
        shoreline = os.path.join(filepath, sitename, sitename + '_reference_shoreline.geojson')
    if out is None:
        out = os.path.join(filepath, sitename, sitename + '_transects.geojson') if sitename else 'transects.geojson'

    lines, epsg = load_shoreline(shoreline, epsg)
    segments = [seg for xy in lines for seg in split_at_gaps(xy)]
    transects = make_transects(segments, spacing)
    if not transects:
        raise ValueError('no transects made, is the shoreline shorter than the spacing?')
    # names: prefix='TA' gives TA1, TA2... (default: GG0001_1, GG0001_2...)
    if prefix is None:
        prefix = f'{sitename}_' if sitename else 'T'
    for n, t in enumerate(transects, 1):
        t['name'] = f'{prefix}{n}'
    print(f'{len(transects)} transects on {len(segments)} segment(s)')

    def ends(t):   # [land point, sea point]
        return [t['centre'] - t['normal'] * land_length, t['centre'] + t['normal'] * sea_length]

    # which side is the sea?
    if sea_side == 'auto':
        print('Finding the sea side with MNDWI...')
        votes = []
        for img in clear_images(sitename, filepath):
            votes.append(votes_on_image(transects, img, epsg, land_length, sea_length))
            print(f"  using {img['sat']} {img['date']:%Y-%m-%d}")
        if not votes:
            raise RuntimeError('no clear image found, use --sea-side left or right instead')
        orient(transects, votes)
    elif sea_side == 'left':
        for t in transects:
            t['normal'] = -t['normal']

    # flag transects that cross their neighbour (tight curves, harbours)
    for a, b in zip(transects, transects[1:]):
        if a['segment'] == b['segment'] and LineString(ends(a)).intersects(LineString(ends(b))):
            a['check'] = b['check'] = True

    gdf = gpd.GeoDataFrame({'name': [t['name'] for t in transects],
                            'check': [bool(t['check']) for t in transects]},
                           geometry=[LineString(ends(t)) for t in transects], crs=epsg)
    gdf.to_file(out, driver='GeoJSON')
    # figure with CoastSat's mapped shorelines (<sitename>_output.pkl), if they exist
    fn = os.path.join(filepath, sitename or '', f'{sitename}_output.pkl')
    if output is None and sitename and os.path.exists(fn):
        with open(fn, 'rb') as f:
            output = pickle.load(f)
    plot_qc(transects, ends, segments, out.replace('.geojson', '_qc.jpg'), output)
    print(f'Saved {out} ({int(gdf["check"].sum())} transects to check)')
    return {t['name']: np.array(ends(t)) for t in transects}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Auto-generate shore-normal transects for CoastSat.')
    p.add_argument('--sitename', help='CoastSat site name')
    p.add_argument('--filepath', default='data', help='CoastSat data folder')
    p.add_argument('--shoreline', help='reference shoreline (.geojson or .pkl), default: the one CoastSat saved')
    p.add_argument('--epsg', type=int, help='output EPSG code (needed for a .pkl)')
    p.add_argument('--sea-side', default='auto', choices=['auto', 'left', 'right'],
                   help='left/right of the direction the shoreline was drawn, or auto')
    p.add_argument('--spacing', type=float, default=100, help='distance between transects (m)')
    p.add_argument('--land-length', type=float, default=200, help='length on the land side (m)')
    p.add_argument('--sea-length', type=float, default=200, help='length on the sea side (m)')
    p.add_argument('--out', help='output .geojson')
    p.add_argument('--prefix', help='transect names, e.g. TA gives TA1, TA2...')
    a = p.parse_args()
    auto_generate_transects(a.shoreline, a.epsg, a.sitename, a.filepath, a.sea_side,
                            a.spacing, a.land_length, a.sea_length, a.out, a.prefix)
