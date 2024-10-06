# FES2022 SETUP INSTRUCTIONS

This document shows how to setup the FES2022 global tide model to get modelled tides at a given location for a given time period.

## 1. Install CoastSat (starting by pyfes)
If you have already installed `coastsat` before, create a new environment with a new name, as you will need to install `pyfes` first otherwise there are conflicts with the python version.

1. Download Anaconda for your operating system https://docs.anaconda.com/anaconda/install/windows/
2. Open the Anaconda Prompt as Administrator and type the following commands:
```
conda clean --all -y
conda update -n base -c conda-forge conda -y
conda create -n coastsat
conda activate coastsat
conda install fbriol::pyfes -y
conda install -c conda-forge geopandas -y
conda install -c conda-forge earthengine-api scikit-image matplotlib astropy notebook -y
pip install pyqt5 imageio-ffmpeg
```
Now you have setup the `coastsat` python environment, which can be activated with `conda activate coastsat`.

Either clone the Github repository https://github.com/kvos/CoastSat or download the zip folder and unzip in a local folder.

## 2. Download FES2022 netcdf files

1. Go to [this location](https://unsw-my.sharepoint.com/personal/z2273773_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz2273773%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FRESEARCH2%2FFES%202022%2Ffes2022b&ga=1) and download the all the files and put them in a folder. These are the netcdf file containing the tidal constituents for the whole world. 
    - If you can't access this link, you will need to download the files yourself from the [AVISO website](https://www.aviso.altimetry.fr/) and follow the steps below. 
    - If you can access the link, skip to **step 7**.

2. Go to https://www.aviso.altimetry.fr/ and create an account, then login. Then go to https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html and fill the form, ticking **FES (Finite Element Solution - Oceanic Tides Heights)**.

3. Navigate to My Products (https://www.aviso.altimetry.fr/en/my-aviso-plus/my-products.html) and the FES product should be there as shown here: ![image](https://github.com/user-attachments/assets/88ffd3ea-ee91-4faa-96e2-fe8982290843)

4. Download [WinSCP](https://winscp.net/eng/download.php) or your favourite SFTP software and click use SFTP link (sftp://ftp-access.aviso.altimetry.fr:2221/auxiliary/tide_model) to create a connection.

5. Then download under /fes2022b the folders /load_tide and /ocean_tide (not /ocean_tide_extrapolate). Download all the components (34 NETCDF files) and unzip them. ![image](https://github.com/user-attachments/assets/39c00bf6-2949-4321-83ed-03b11a39c0b7)

6. Finally download the `fes2022.yaml` from https://github.com/CNES/aviso-fes/tree/main/data/fes2022b and save it in the same folder as /load_tide and /ocean_tide. 

7. Open the `fes2022.yaml` file in a text editor and change the path to each of the tidal constituents (individual netcdf files). Add the absolute path to each .nc file, an example is shown below. You can use find and replace to do this in one go. It should look like below:
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

    Your Python environment can map shorelines and predict tides anywhere in the world.

## Test that it's working

To test your installation, open the Anaconda Prompt.
Activate the coastsat environment and open Python:
- `conda activate coastsat`
- `python`
Locate the path to your `fes2022.yaml` file and copy it. Then type:
- `import pyfes`
- `filepath = PATH_TO_fes2022.yaml`
- `handlers = pyfes.load_config(config)`

This last command may take 5 minutes to run but if it doesn't return an error you are all good to go.

You can now generate tide time-series using FES2022 for any location and any dates. 

## Example
```
# load pyfes and the global tide model (may take one minute)
import pyfes
filepath = os.path.join(os.pardir,'CoastSat.webgis','aviso-fes-main','data','fes2022b')
config =  os.path.join(filepath, 'fes2022.yaml')
handlers = pyfes.load_config(config)
ocean_tide = handlers['tide']
load_tide = handlers['radial']
# load coastsat module to estimate slopes
from coastsat import SDS_slopes

# get centroid, date range and timestep
centroid = [151.3023463 -33.7239154]
date_range = [pytz.utc.localize(datetime(2024,1,1)),
              pytz.utc.localize(datetime(2025,1,1))]
timestep = 900 # in seconds

# predict tides
dates_ts, tides_ts = SDS_slopes.compute_tide(centroid, date_range, timestep, ocean_tide, load_tide)
```