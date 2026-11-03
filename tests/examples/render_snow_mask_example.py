"""Regenerate the Nome comparison image from the bundled scene crop."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coastsat.SDS_preprocess import create_sentinel_snow_mask


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'nome-l1c-validation.npz'
OUTPUT = HERE / 'snow-mask-alaska-west-nome.png'


def main():
    with np.load(FIXTURE, allow_pickle=False) as data:
        green_20m = data['green_20m'].astype(np.float32) / 10000
        swir1_20m = data['swir1_20m'].astype(np.float32) / 10000
        scl_20m = data['scl_20m']
        rgb_10m = data['rgb_tci_10m']

    valid = (scl_20m != 0) & (scl_20m != 1)
    ndsi_mask = create_sentinel_snow_mask(green_20m, swir1_20m) & valid
    scl_mask = (scl_20m == 11) & valid
    ndsi_10m = np.repeat(np.repeat(ndsi_mask, 2, axis=0), 2, axis=1)
    scl_10m = np.repeat(np.repeat(scl_mask, 2, axis=0), 2, axis=1)
    ndsi_10m = ndsi_10m[:rgb_10m.shape[0], :rgb_10m.shape[1]]
    scl_10m = scl_10m[:rgb_10m.shape[0], :rgb_10m.shape[1]]

    intersection = np.logical_and(ndsi_mask, scl_mask).sum()
    union = np.logical_or(ndsi_mask, scl_mask).sum()
    ndsi_coverage = ndsi_mask.sum() / valid.sum() * 100
    scl_coverage = scl_mask.sum() / valid.sum() * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), layout='constrained')
    axes[0].imshow(rgb_10m)
    axes[0].set_title('True colour, no snow mask')

    for ax, mask, title in (
            (axes[1], ndsi_10m, 'Level-1C NDSI snow/ice mask'),
            (axes[2], scl_10m, 'Level-2A SCL class 11, comparison only')):
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (1.0, 0.10, 0.12, 0.55)
        ax.imshow(rgb_10m)
        ax.imshow(overlay)
        ax.set_title(title)

    for ax in axes:
        ax.axis('off')

    fig.suptitle('Natural Arctic coast west of Nome, Alaska, Sentinel-2A, 1 June 2024')
    fig.text(
        0.5, 0.01,
        f'NDSI coverage: {ndsi_coverage:.1f}% | '
        f'SCL reference: {scl_coverage:.1f}% | '
        f'Intersection over union: {intersection / union:.2f}. '
        'SCL is included only as a visual reference.',
        ha='center')
    fig.savefig(OUTPUT, dpi=150)
    plt.close(fig)
    print(f'Wrote {OUTPUT}')
    print(f'NDSI={ndsi_coverage:.1f}%, SCL={scl_coverage:.1f}%, IoU={intersection / union:.2f}')


if __name__ == '__main__':
    main()
