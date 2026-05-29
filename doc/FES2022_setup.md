# FES2022 SETUP INSTRUCTIONS

This document shows how to setup the FES2022 global tide model to get modelled tides at a given location for a given time period.

## 1. Install pyfes

To install the `pyfes` package, activate your `coastsat` environment and run the following command:
```
conda install -c conda-forge pyfes
```
:warning: if a conflict error occurs, try installing with `pip` following the instructions at https://github.com/CNES/aviso-fes. Otherwise re-install the full `coastsat` environment with the `pyfes` package included, often the easier solution.

## 2. Download FES2022 netcdf files

1. Go to https://www.aviso.altimetry.fr/ and create an account, then login. Then go to https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html and fill the form, ticking **FES (Finite Element Solution - Oceanic Tides Heights)**.

2. Login and navigate to My Products (https://www.aviso.altimetry.fr/en/my-aviso-plus/my-products.html) and the FES product should be there as shown here: ![image](https://github.com/user-attachments/assets/88ffd3ea-ee91-4faa-96e2-fe8982290843)

3. Download [WinSCP](https://winscp.net/eng/download.php) or your favourite SFTP software and click on `Sftp` to create a connection.

4. Then download under /fes2022b the folders /load_tide and /ocean_tide (do not download /ocean_tide_extrapolate). Download all the components (34 NETCDF files) and unzip them. ![image](https://github.com/user-attachments/assets/39c00bf6-2949-4321-83ed-03b11a39c0b7)

5. Finally download the `fes2022.yaml` from https://github.com/CNES/aviso-fes/tree/main/data/fes2022b and save it in the same folder as /load_tide and /ocean_tide. 

6. Open the `fes2022.yaml` file in a text editor and change the path to each of the tidal constituents (individual netcdf files). Add the absolute path to each .nc file (pay attention to special characters), an example is shown below. You can use find and replace to do this in one go. It should look like below:
    ```
    radial:
    cartesian:
        amplitude: amplitude
        latitude: lat
        longitude: lon
        paths:
        2N2: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\2n2_fes2022.nc
        Eps2: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\eps2_fes2022.nc
        J1: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\j1_fes2022.nc
        K1: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\k1_fes2022.nc
        K2: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\k2_fes2022.nc
        L2: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\l2_fes2022.nc
        Lambda2: C:\Users\kilia\Documents\GitHub\CoastSat\fes2022b\load_tide\lambda2_fes2022.nc
    etc...
    ```
    Make sure to do this for both `radial` and `tide` parts of the file.

Your Python environment can now map shorelines and predict tides anywhere in the world using FES2022.

## 3. Test that it's working

To test your installation, open the Miniforge/Anaconda Prompt.
Activate the coastsat environment and open Python:
- `conda activate coastsat`
- `python`
Locate the path to your `fes2022.yaml` file and copy it. Then type:
- `import pyfes`
- `config_file = LOCAL_PATH/fes2022.yaml` (for LOCAL_PATH copy the location of the directory that contains `fes2022.yaml`)
- `config = pyfes.config.load(config_file)` this command loads the netcdf files in memory and requires >10GB RAM, if your machine cannot handle this, use the script [clip_tide_netcdf_by_latitude.py](./examples/tide_model_clipping/clip_tide_netcdf_by_latitude.py) to clip the netcdf files to a smaller region of interest and reduce their size in memory.

If the command works without error, you are ready to go!

## 4. Example of tide prediction
```python
import pyfes
from coastsat import SDS_slope
from datetime import datetime
import pytz

# example centroid locations to test
centroid = (151.309093, 33.716037) # narrabeen
centroid = (-1.281540, 44.726732) # truc vert
centroid = (-75.7252, 36.1696) # duck

# if you have clipped the netcdf files by latitude
with open(os.path.join(output_dir, "band_configs.json"), "r") as f:
    band_configs = json.load(f)
selected_band = SDS_slope.select_yaml_for_centroid(centroid, band_configs)
selected_yaml = selected_band["yaml"]

# if you want to load all netcdf files (requires >10GB RAM)
# selected_yaml = 'LOCAL_PATH/fes2022.yaml' # replace LOCAL_PATH

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
print('Validation value = 0.29')
```
