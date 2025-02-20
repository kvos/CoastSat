import os
from glob import glob
# load coastsat module located two folders up
import sys
sys.path.insert(0, os.path.join(os.getcwd(), os.pardir, os.pardir))
from coastsat import SDS_slope

"""
clip_tide_files.py
________________________________________________________

This script is meant to clip the fes2022 tide model area
to allow for less resource consumption.

The suggested approach is to trim the area to a polygon
containing anywhere you may want to run CoastSat, and
you will need a geojson file containing the region.
It is best to leave it with some room offshore, to allow
for any interpolation the model may need to do. You will
need to switch the fes2022.yaml to the new one created
to make use of the clipped files when running CoastSat.

PS 2025

If you have any questions about how this script is used,
don't hesitate to mention me in a github issue.
"""


# 1) Define inputs
# Path to geojson file, note this file should contain any area you may want to run coastsat in
# Ex: The whole of Europe or the whole of Australia, or the whole of Canada
geojson_file = "Europe.geojson" #r"C:\path\to\example\region.geojson"
# Path to existing 'load_tide' directory
load_tide_dir = r"C:\path\to\load_tide"   # 'radial' dataset
# Path to existing 'ocean_tide' directory
ocean_tide_dir = r"C:\path\to\ocean_tide" # 'tide' dataset
# Where the new clipped files will be put
output_dir = os.getcwd()

# Path to original fes2022.yaml file
original_yaml = r"C:\path\to\existing\fes2022.yaml"
# Where the new fes2022_clipped.yaml will be put
new_yaml = os.path.join(os.getcwd(), "fes2022_clipped.yaml")

# 2) Load the region geometry
geometry = SDS_slope.get_region_from_geojson(geojson_file)

# 3) Find input NetCDF files
load_tide_files = glob(os.path.join(load_tide_dir, "*.nc"))
ocean_tide_files = glob(os.path.join(ocean_tide_dir, "*.nc"))

# 4) Clip the files
print("Clipping load_tide (radial) files...")
radial_map = SDS_slope.clip_model_to_region(
    load_tide_files,
    geometry,
    os.path.join(output_dir, "load_tide")
)

print("Clipping ocean_tide (tide) files...")
tide_map = SDS_slope.clip_model_to_region(
    ocean_tide_files,
    geometry,
    os.path.join(output_dir, "ocean_tide")
)

# 5) Create a dict for the new paths
path_maps = {
    "radial": radial_map,
    "tide": tide_map
}

# 6) Create a new YAML with the updated paths
SDS_slope.write_new_fes_yaml(original_yaml, new_yaml, path_maps)
