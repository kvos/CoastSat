# CoastSat

CoastSat is an open-source software toolkit written in Python that enables users to obtain time-series of shoreline position at any coastline worldwide from 30+ years (and growing) of publicly available satellite imagery.

![Alt text](https://github.com/kvos/CoastSat/blob/development/classifiers/doc/example.gif)

The underlying approach and application of the CoastSat toolkit are described in detail in:

*Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (submitted). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery, Environmental Modelling and Software*.

There are two main steps:
- assisted retrieval from from Google Earth Engine of all avaiable satellite images spanning the user-defined region of interest and time period
- automated extraction of shorelines from all the selected images using a sub-pixel resolution technique


### Description

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ field measurements are available. Satellite imagery spanning the last 30 years with constant revisit periods is publicly available and suitable to extract repeated measurements of the shoreline position.
CoastSat enables the non-expert user to extract shorelines from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 images.
The shoreline detection algorithm implemented in CoastSat is optimised for sandy beach coastlines.  It combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

## 1. Installation

CoastSat requires the following Python packages to run:
```
conda-forge: python=3.6 | matplotlib | scikit-image | scikit-learn | gdal | earthengine-api | oauth2client | spyder | jupyter | simplekml
PyPi:        shapely
```
If you are not a regular Python user and are not sure how to install these packages from *conda-forge* and *PyPi*, the section below explains how to install them step-by-step using Anaconda. More experinced Python users can proceed to install these packages and go directly to section **1.2 Activating Google Earth Engine Python API**.

### 1.1 Installing the packages (Anaconda)

If Anaconda is not already installed on your PC, it can be freely downloaded at https://www.anaconda.com/download/.
Open the *Anaconda prompt* (in Mac and Linux, open a terminal window) and select the folder where you have downloaded/cloned this repository.

Create a new environment named *coastsat*:

```
conda create -n coastsat
```

Activate the new environment:

```
conda activate coastsat
```

On Linux systems, type `source activate coastsat` instead.

To confrim that you have successfully activated CoastSat, your terminal command line prompt should now start with (coastsat).

Now you need to populate the environment with the packages needed to run the CoastSat toolkit. All the necessary packages are contained in three platform specific files: `requirements_win64.txt`, `requirements_osx64.txt`, `requirements_linux64.txt`. To install the package for your pc platform, run one of the following commands, depending on which platform you are operating:

#### Windows 64 bits (win64)

```
conda install --name coastsat --file requirements_win64.txt
```

#### Mac 64 bits (osx64)

```
conda install --name coastsat --file requirements_osx64.txt
```

#### Linux 64 bits (linux64)

```
conda install --name coastsat --file requirements_linux64.txt
```

This might take a few minutes... once it is finished, run the following command:

```
pip install shapely
```

All the packages have now been install in an environment called `coastsat`.

### 1.2 Activating Google Earth Engine Python API

Go to https://earthengine.google.com and sign up to Google Earth Engine (GEE).

![gee_capture](https://user-images.githubusercontent.com/7217258/49348457-a9271300-f6f9-11e8-8c0b-407383940e94.jpg)

Once you have created a Google Earth Engine account, go back to Anaconda and link your GEE credentials to the Python API:

```
earthengine authenticate
```

A web browser will open. Login with your GEE credentials, read and accept the terms and conditions, and copy the authorization code into the Anaconda terminal.

Now you are ready to start using the CoastSat toolbox!

## 2. Usage

**Note**: remeber to always activate the `coastsat` environment with `conda activate coastsat` each time you are preparing to use it.
Your terminal command line prompt should always start with (coastsat) to confirm that it is activated.

An example of how to run the software in a Jupyter Notebook is provided in the repository (`example_jupyter.ipynb`). To run this, first activate your `coastsat` environment with `conda activate coastsat` (if not already active), and then type:

```
jupyter notebook
```

A web browser window will open, drive to the directory where you downloaded/cloned this repository and click on `example_jupyter.ipynb`.
The following sections guide the reader through the different functionalities of CoastSat with an example at Narrabeen beach (Australia). If you prefer to use Spyder or PyCharm or other integrated development environments (IDEs), a Python script is also included `main.py` in the repository.

To run a Jupyter Notebook, place your cursor inside one of the code sections and then clikc on the 'run' button up in the top menu to run that section and progress forward (as shown in the animation below).

![example_jupyter](https://user-images.githubusercontent.com/7217258/49705486-8dc88480-fc72-11e8-8300-c342baaf54eb.gif)

### 2.1 Retrieval of the satellite images

To retrieve from the GEE server the avaiable satellite images cropped around the required region of coasltine for the particular time period of interest, the following user-defined variables are required:
- `polygon`: the coordinates of the region of interest (longitude/latitude pairs)
- `dates`: dates over which the images will be retrieved (e.g., `dates = ['2017-12-01', '2018-01-01']`)
- `sat_list`: satellite missions to consider (e.g., `sat_list = ['L5', 'L7', 'L8', 'S2']` for Landsat 5, 7, 8 and Sentinel-2 collections)
- `sitename`: name of the site (user-defined name of the subfolder where the images and other accompanying files will be stored)

The call `metadata = SDS_download.retrieve_images(inputs)` will launch the retrieval of the images and store them as .TIF files (under *.data\sitename*). The metadata contains the exact time of acquisition (UTC) and geometric accuracy of each downloaded image and is saved as `metadata_sitename.pkl`. If the images have already been downloaded previously and the user only wants to run the shoreline detection, the metadata can be loaded directly from this file. The screenshot below shows an example where all the images of Collaroy-Narrrabeen (Australia) acquired in December 2017 are retrieved.

![retrieval](https://user-images.githubusercontent.com/7217258/49353105-0037e280-f710-11e8-9454-c03ce6116c54.PNG)

### 2.2 Shoreline detection

It is now time to map the sandy shorelines!

The following user-defined settings are required:
- `cloud_thresh`: threshold on maximum cloud cover that is acceptable on the images (value between 0 and 1 - this may require some initial experimentation)
- `output_epsg`: epsg code defining the spatial reference system of the shoreline coordinates
- `check_detection`: if set to `True` allows the user to quality control each shoreline detection

See http://spatialreference.org/ to find the EPSG number corresponding to your local coordinate system. If the user wants to quality control the mapped shorelines and manually validate each detection, the parameter `check_detection` should be set to `True`.

In addition, there are extra parameters (`min_beach_size`, `buffer_size`, `min_length_sl`) that can be tuned to optimise the shoreline detection (for Advanced users only). For the moment leave these parameters set to their default values, we will see later how they can be modified.

An example of settings is provided here:

![settings](https://user-images.githubusercontent.com/7217258/49565578-ba7f5200-f97b-11e8-9bb4-8d933329b625.PNG)

Once all the settings have been defined, the batch shoreline detection can be launched by calling:
```
output = SDS_shoreline.extract_shorelines(metadata, settings)
```
When `check_detection` is set to `True`, a figure like the one below appears and asks the user to manually accept/reject each detection by clicking on `keep` or `skip`.

![Alt text](https://github.com/kvos/CoastSat/blob/development/classifiers/doc/batch_detection.gif)

Once all the shorelines have been mapped, the output is available in two different formats (saved under *.\data\sitename*):
- `sitename_output.pkl`: contains a list with the shoreline coordinates and the exact timestamp at which the image was captured (UTC time) as well as the geometric accuracy and the cloud cover of each indivdual image. This list can be manipulated with Python, a snippet of code to plot the results is provided in the main script.
- `sitename_output.kml`: this output can be visualised in a GIS software (e.g., QGIS, ArcGIS).

The figure below shows how the satellite-derived shorelines can be opened in a GIS software (QGIS) using the `.kml` output.

![gis_output](https://user-images.githubusercontent.com/7217258/49361401-15bd0480-f730-11e8-88a8-a127f87ca64a.jpeg)

#### Advanced shoreline detection parameters

As mentioned above, there are some additional parameters that can be modified to optimise the shoreline detection:
- `min_beach_area`: minimum allowable object area (in metres^2) for the class 'sand'. During the image classification, some features (for example, building roofs) may be incorrectly labelled as sand. To correct this, all the objects classified as sand containing less than a certain number of connected pixels are removed from the sand class. The default value of `min_beach_area` is 4500 m^2, which corresponds to 20 connected pixels of 15 m^2. If you are looking at a very small beach (<20 connected pixels on the images), try decreasing the value of this parameter.
- `buffer_size`: radius (in metres) that defines the buffer around sandy pixels that is considered for the shoreline detection. The default value of `buffer_size` is 150 m. This parameter should be increased if you have a very wide (>150 m) surf zone or inter-tidal zone.
- `min_length_sl`: minimum length (in metres) of shoreline perimeter to be valid. This can be used to discard small features that are detected but do not correspond to the sand-water shoreline. The default value is 200 m. If the shoreline that you are trying to map is shorter than 200 m, decrease the value of this parameter.

#### Reference shoreline

There is also an option to manually digitize a reference shoreline before running the batch shoreline detection on all the images. This reference shoreline helps to reject outliers and false detections when mapping shorelines as it only considers as valid shorelines the points that are within a distance from this reference shoreline.

 The user can manually digitize a reference shoreline on one of the images by calling:
```
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_manual(metadata, settings)
settings['max_dist_ref'] = 100 # max distance (in meters) allowed from the reference shoreline
```
This function allows the user to click points along the shoreline on one of the satellite images, as shown in the animation below.

![ref_shoreline](https://user-images.githubusercontent.com/7217258/49710753-94b1c000-fc8f-11e8-9b6c-b5e96aadc5c9.gif)

The maximum distance (in metres) allowed from the reference shoreline is defined by the parameter `max_dist_ref`. This parameter is set to a default value of 100 m. If you think that your shoreline will move more than 100 m, please change this parameter to an appropriate distance. This may be the case for large nourishments or eroding/accreting coastlines.

### 2.3 Shoreline change analysis

This section shows how to obtain time-series of shoreline change along shore-normal transects.

The user can draw shore-normal transects by calling:
```
settings['transect_length'] = 500 # defines the length of the transects in metres
transects = SDS_transects.draw_transects(output, settings)
```
Once the shore-normal transects have been defined, the intersection between the 2D shorelines and the transects is computed with the following function:
```
settings['along_dist'] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings)
```
The parameter `along_dist` defines the along-shore distance around the transect over which shoreline points are selected to compute the intersection. The default value is 25 m, which means that the intersection is computed as the median of the points located within 25 m of the transect (50 m alongshore-median).

An example is illustrated below:

![transects](https://user-images.githubusercontent.com/7217258/49990925-8b985a00-ffd3-11e8-8c54-57e4bf8082dd.gif)


## Issues and Contributions

Having a problem or looking to contribute to the code? Please see the [Issues page](https://github.com/kvos/coastsat/issues).
