#==========================================================#
# Clip FES2022 Tide Files
#==========================================================#

# PS 2025

#%% 1. Initial settings

import os
import json
from datetime import datetime
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyfes
import pytz

from coastsat import SDS_slope

matplotlib.use("Agg")
plt.ion()

# Path to existing 'fes2022' directory with 'ocean_tide' and 'load_tide' subdirs
fp_fes2022 = r"C:\Users\KilianVos\Documents\fes2022b"
fp_ocean_tide =  os.path.join(fp_fes2022, 'ocean_tide')
fp_load_tide = os.path.join(fp_fes2022, 'load_tide')
original_yaml = os.path.join(fp_fes2022,'fes2022.yaml')
if not os.path.exists(fp_ocean_tide) or not os.path.exists(fp_load_tide) or not os.path.exists(original_yaml):
    raise FileNotFoundError("Could not find the original FES2022 files. Please check the path settings.")

# Create new output directory for clipped files
output_dir = r"C:\Users\KilianVos\Documents\GitHub\CoastSat\examples\tide_model_clipping\fes2022_by_latitude"
os.makedirs(output_dir, exist_ok=True)

#%% 2. Clip original files by 20 degree latitude bands
band_height, lat_min, lat_max = 20, -80, 80
latitude_bands = SDS_slope.build_latitude_bands(lat_min, lat_max, band_height)

# loop through bands and clip files, saving new YAML configs for each band (this could take 5-10 minutes)
band_configs = []
ocean_tide_files = glob(os.path.join(fp_ocean_tide, "*.nc"))
load_tide_files = glob(os.path.join(fp_load_tide, "*.nc"))
for band_lat_min, band_lat_max in latitude_bands:
    band_tag = SDS_slope.format_lat_band_tag(band_lat_min, band_lat_max)
    load_tide_output_dir = os.path.join(output_dir, f"load_tide_{band_tag}")
    ocean_tide_output_dir = os.path.join(output_dir, f"ocean_tide_{band_tag}")
    band_yaml = os.path.join(output_dir, f"fes2022_{band_tag}.yaml")
    radial_map = SDS_slope.clip_model_to_region(load_tide_files,band_lat_min,band_lat_max,load_tide_output_dir)
    tide_map = SDS_slope.clip_model_to_region(ocean_tide_files,band_lat_min,band_lat_max,ocean_tide_output_dir)
    path_maps = {"radial": radial_map,"tide": tide_map}
    SDS_slope.write_new_fes_yaml(original_yaml, band_yaml, path_maps)
    band_configs.append({
        "tag": band_tag,
        "lat_min": band_lat_min,
        "lat_max": band_lat_max,
        "ocean_tide_dir": ocean_tide_output_dir,
        "load_tide_dir": load_tide_output_dir,
        "yaml": band_yaml,
    })
# save band configs
with open(os.path.join(output_dir, "band_configs.json"), "w") as f:
    json.dump(band_configs, f, indent=4)

#%% 3. (Optional) Visualise springs tide range for original and clipped files

# plot the global tidal range map
grid_coords, amplitude_springs = SDS_slope.get_springs_tide_range(fp_ocean_tide)
fp_fig = os.path.join(output_dir, "tide_map.png")
SDS_slope.plot_tide_map(grid_coords, amplitude_springs, fp_fig, decimate=100)

# then plot the clipped files to check they look correct
for band_config in band_configs:
    grid_coords_clipped, amplitude_springs_clipped = SDS_slope.get_springs_tide_range(band_config["ocean_tide_dir"])
    fp_fig_clipped = os.path.join(output_dir, f"tide_map_clipped_{band_config['tag']}.png")
    SDS_slope.plot_tide_map(grid_coords_clipped, amplitude_springs_clipped, fp_fig_clipped, decimate=100)

#%% 4. Evaluate tide levels at a test point to check files work with pyfes

centroid = (151.309093, 33.716037) # narrabeen
centroid = (-1.281540, 44.726732) # truc vert
centroid = (-75.7252, 36.1696) # duck
    
# select the correct YAML config based on the centroid latitude
with open(os.path.join(output_dir, "band_configs.json"), "r") as f:
    band_configs = json.load(f)
selected_band = SDS_slope.select_yaml_for_centroid(centroid, band_configs)
selected_yaml = selected_band["yaml"]

# load files using pyfes
if int(pyfes.__version__.split('.')[0]) < 2026:
    handlers = pyfes.load_config(selected_yaml)
    ocean_tide = handlers['tide']
    load_tide = handlers['radial']
else:
    config = pyfes.config.load(selected_yaml)
    ocean_tide = config.models['tide']
    load_tide = config.models['radial']

# remove negative longitudes
if centroid[0] < 0:
    centroid = (centroid[0]+360, centroid[1])

# duck test
date = pytz.utc.localize(datetime(2025,1,1,0,15,0))
tide_level = SDS_slope.compute_tide_dates(centroid,[date],ocean_tide,load_tide)
print("Tide level at {} on {} is {}".format(centroid, date, tide_level))
print('Validation value = 0.30')