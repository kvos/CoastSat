# WORK IN PROGRESS, please do not review yet!

# CoastSat

Python code to extract shorelines at sub-pixel resolution from publicly available satellite imagery. The shoreline detection algorithm is described in *Vos K., Harley M.D., Splinter K.D., Simmons J.A., Turner I.L. (in review). Capturing intra-annual to multi-decadal shoreline variability from publicly available satellite imagery, Coastal Engineering*. Google Earth Engine's Python API is used to access the archive of publicly available satellite imagery (Landsat series and Sentinel-2).

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ measurements are available. Satellite imagery spannig the last 30 years with constant revisit periods is publicly available and suitable to extract repeated measurements of the shoreline positon.
*coastsat* is an open-source Python module that allows to extract shorelines from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 images.
The shoreline detection algorithm proposed here combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

## 1. Installation

CoastSat requires the following Python packages to run: 
```
python=3.6 | matplotlib | scikit-image | scikit-learn | gdal | earthengine-api | oauth2client | spyder | jupyter | shapely | simplekml
```
If you are not a regular Python user and are not sure how to install these packages, the section below shows how to install them using Anaconda on Windows (x64). Otherwise, install the packages and go directly to section **1.2 Activating Google Earth Engine Python API**.

### 1.1 Installing the packages (Anaconda)

If Anaconda is not already installed on your PC, you can get it at https://www.anaconda.com/download/.
Open the *Anaconda prompt* and drive to the folder where you downloaded/cloned this repository. There are two ways of cloning an environment with Anaconda, try **Option 1** first and if this doesn't work, try **Option 2**. 

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

Now you are ready to start using the toolbox!

## Usage 

A demonstration of the use of *coastsat* is provided in the Jupyter Notebook *shoreline_extraction.ipynb*. The code can also be run in Spyder with *main_spyder.py*.

The first step is to retrieve the satellite images of the region of interest from Google Earth Engine servers by calling *SDS_download.get_images(sitename, polygon, dates, sat_list)*:
- *sitename* is a string which will define the name of the folder where the files will be stored
- *polygon* contains the coordinates of the region of interest (longitude/latitude pairs)
- *dates* defines the dates over which the images will be retrieved (e.g., *dates = ['2017-12-01', '2018-01-01']*)  
- *sat_list* indicates which satellite missions to consider (e.g., *sat_list = ['L5', 'L7', 'L8', 'S2']* will download images from Landsat 5, 7, 8 and Sentinel-2 collections).

The images are cropped on the Google Earth Engine servers and only the region of interest is downloaded resulting in low memory allocation (~ 1 megabyte/image for a 5km-long beach). The relevant image metadata (time of acquisition, geometric accuracy...etc) is stored in a file named *sitename_metadata.pkl*.

Once the images have been downloaded, the shorelines are extracted from the multispectral images using the sub-pixel resolution technique described in *Vos K., Harley M.D., Splinter K.D., Simmons J.A., Turner I.L. (in review). Capturing intra-annual to multi-decadal shoreline variability from publicly available satellite imagery, Coastal Engineering*.
The shoreline extraction is performed by the function SDS_shoreline.extract_shorelines(metadata, settings). The user must define the settings in a Python dictionary. To ensure maximum robustness of the algorithm the user can optionally digitize a reference shoreline (byc calling *SDS_preprocess.get_reference_sl(metadata, settings)*) that will then be used to identify obvious outliers and minimize false detections. Since the cloud mask is not perfect (especially in Sentinel-2 images) the user has the option to manually validate each detection by setting the *'check_detection'* parameter to *True*.
The shoreline coordinates (in the coordinate system defined by the user in *'output_epsg'* are stored in a file named *sitename_out.pkl*.

## Issues and Contributions

Looking to contribute to the code? Please see the [Issues page](https://github.com/kvos/coastsat/issues).
