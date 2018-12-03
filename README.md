# CoastSat

This software enables the users to extract time-series of shoreline change over the last 30+ years at their site of interest. The software is described in *Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (submitted). CoastSat: a Google Earth Engine-enabled software to extract shorelines from publicly available satellite imagery, Environmental Modelling and Software*. 

There are two main steps:
- retrieval of the satellite images of the region of interest from Google Earth Engine
- extraction of the shorelines from the images using a sub-pixel resolution technique

### Description

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ measurements are available. Satellite imagery spannig the last 30 years with constant revisit periods is publicly available and suitable to extract repeated measurements of the shoreline positon.
CoastSat is an open-source Python module that allows to extract shorelines from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 images.
The shoreline detection algorithm implemented in CoastSat combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

## 1. Installation

CoastSat requires the following Python packages to run: 
```
python=3.6 | matplotlib | scikit-image | scikit-learn | gdal | earthengine-api | oauth2client | spyder | jupyter | shapely | simplekml
```
If you are not a regular Python user and are not sure how to install these packages, the section below shows how to install them using Anaconda. Otherwise, install the packages and go directly to section **1.2 Activating Google Earth Engine Python API**.

### 1.1 Installing the packages (Anaconda)

If Anaconda is not already installed on your PC, you can get it at https://www.anaconda.com/download/.
Open the *Anaconda prompt* and drive to the folder where you downloaded/cloned this repository. There are two ways of cloning an environment with Anaconda, try **Option 1** first and if the installation fails, try **Option 2** (only for Windows x64). 

#### Option 1

Create an environment with all the necessary Python packages by running the following command:

```
conda env create -f environment.yml
```

#### Option 2


Create a new environment named *coastsat* with Python 3.6: 

```
conda create -n coastsat python=3.6
```

Activate the new environment:

```
conda activate coastsat
```

Now populate the environment with the packages needed to run the toolbox. All the necessary packages are contained in the **requirements.txt** file. Run this command to install them on Windows 64 bits (this might take a few minutes):

```
conda install --name coastsat --file requirements.txt
```

### 1.2 Activating Google Earth Engine Python API

Go to https://earthengine.google.com and sign up to Google Earth Engine.

![gee_capture](https://user-images.githubusercontent.com/7217258/49348457-a9271300-f6f9-11e8-8c0b-407383940e94.jpg)

Once you have created a Google Earth Engine account, go back to Anaconda and install Google Earth Engine's Python API package:

```
conda install -c conda-forge earthengine-api
```

Link your GEE credentials to the Python API (for this you need oauth2client):

```
pip install oauth2client
earthengine authenticate
```

A web browser will open, login with your GEE credentials and copy the authorization code into the Anaconda terminal.

Now you are ready to start using the CoastSat!

## 2. Usage 

The software can be run from a Jupyter notebook (*main.ipynb*) or from Spyder (*main.py*). In the following sections we describe an example of shoreline detection using the Jupyter notebook.

### 2.1 Retrieval of the satellite images

To retrieve the satellite images cropped around the the region of interest from Google Earth Engine servers the following user-defined variables are needed:
- `polygon`: the coordinates of the region of interest (longitude/latitude pairs)
- `dates`: dates over which the images will be retrieved (e.g., `dates = ['2017-12-01', '2018-01-01']`)  
- `sat_list`: satellite missions to consider (e.g., `sat_list = ['L5', 'L7', 'L8', 'S2']` for Landsat 5, 7, 8 and Sentinel-2 collections).
- `sitename`: name of the site (defines the name of the subfolder where the files will be stored)

The call `metadata = SDS_download.retrieve_images(inputs)` will launch the retrieval of the images and store them as .TIF files (under **.data\sitename**). The metadata contains the exact time of acquisition (UTC) and geometric accuracy of each downloaded image and is saved as `metadata_sitename.pkl`. If the images have already been downloaded previously and the user only wants to run the shoreline detection, the metadata can be loaded directly from this file. The screenshot below shows an example where all the images of Narrabeen-Collaroy (Australia) acquired in December 2017 are retrieved. 

![retrieval](https://user-images.githubusercontent.com/7217258/49353105-0037e280-f710-11e8-9454-c03ce6116c54.PNG)

### 2.2 Shoreline detection

It is finally time to map the shoreline changes at your local beach!  

The following user-defined settings are required:

- `cloud_thresh`: threshold on maximum cloud cover that is acceptable on the images (value between 0 and 1)
- `output_epsg`: epsg code defining the spatial reference system of the shoreline coordinates
- `check_detection`: if set to `True` allows the user to quality control each shoreline detection

See http://spatialreference.org/ to find the EPSG number corresponding to your local coordinate system. An example of settings is provided here:

![settings](https://user-images.githubusercontent.com/7217258/49354499-4b092880-f717-11e8-9877-135393011a48.PNG)

The figure below provides an example of mapped shoreline where the user can manually accept/reject the detection by clicking on `keep` or `skip`.

![output](https://user-images.githubusercontent.com/7217258/49354698-39745080-f718-11e8-878d-266d850519f7.jpg)

The mapped shorelines are provided as two different outputs (saved under *.\data\sitename*):
- `sitename_output.pkl`: contains a list with the shoreline coordinates and the exact timestamp at which the image was captured (UTC time) as well as the geometric accuracy and the cloud cover of the image. The list can be manipulated with Python, a snippet of code to plot the results is provided in the main script.
- `sitename_output.kml`: this output can be visualised in a GIS software (e.g., QGIS, ArcGIS).

The figure below shows how the satellite-derived shorelines can be opened in GIS software using the `.kml` output.

![gis_output](https://user-images.githubusercontent.com/7217258/49361401-15bd0480-f730-11e8-88a8-a127f87ca64a.jpeg)

## Issues and Contributions

Having a problem or looking to contribute to the code? Please see the [Issues page](https://github.com/kvos/coastsat/issues).
