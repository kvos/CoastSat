# CoastSat

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2779293.svg)](https://doi.org/10.5281/zenodo.2779293)
[![Join the chat at https://gitter.im/CoastSat/community](https://badges.gitter.im/spyder-ide/spyder.svg)](https://gitter.im/CoastSat/community)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release](https://img.shields.io/github/release/kvos/CoastSat)](https://GitHub.com/kvos/CoastSat/releases/)

CoastSat is an open-source software toolkit written in Python that enables users to obtain time-series of shoreline position at any coastline worldwide from 30+ years (and growing) of publicly available satellite imagery.

![Alt text](https://github.com/kvos/CoastSat/blob/master/doc/example.gif)

:point_right: Relevant publications:

- Shoreline detection algorithm: https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)
- Accuracy assessment and applications: https://doi.org/10.1016/j.coastaleng.2019.04.004
- Beach slope estimation: https://doi.org/10.1029/2020GL088365 (preprint [here](https://www.essoar.org/doi/10.1002/essoar.10502903.2))
- Satellite-derived shorelines along meso-macrotidal beaches: https://doi.org/10.1016/j.geomorph.2021.107707
- Beach-face slope dataset for Australia: https://doi.org/10.5194/essd-14-1345-2022

:point_right: Other repositories and addons related to this toolbox:
- [CoastSat.slope](https://github.com/kvos/CoastSat.slope): estimates the beach-face slope from the satellite-derived shorelines obtained with CoastSat.
- [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope): shoreline extraction for PlanetScope Dove imagery (near-daily since 2017 at 3m resolution).
- [InletTracker](https://github.com/VHeimhuber/InletTracker): monitoring of intermittent open/close estuary entrances.
- [CoastSat.islands](https://github.com/mcuttler/CoastSat.islands): 2D planform measurements for small reef islands.
- [CoastSeg](https://github.com/dbuscombe-usgs/CoastSeg): image segmentation, deep learning, doodler.
- [CoastSat.Maxar](https://github.com/kvos/CoastSat.Maxar): shoreline extraction on Maxar World-View images (in progress)

:point_right: Visit the [CoastSat website](http://coastsat.wrl.unsw.edu.au/) to explore and download regional-scale datasets of satellite-derived shorelines and beach slopes generated with CoastSat.

:star: **If you like the repo put a star on it!** :star:

### Latest updates

:arrow_forward: *(2022/08/01)*
CoastSat 2.0 (major release):
+ new download function for Landsat images (better alignment between panchromatic and multispectral bands)
+ quality-control steps added for fully automated shoreline extraction
+ post-processing of the shorelne time-series, including despiking and computing seasonal-averages.

:arrow_forward: *(2022/07/20)*
Option to switch off panchromatic sharpening on Landsat 7, 8 and 9 imagery.

:arrow_forward: *(2022/05/02)*
Compatibility with Landsat 9 and Landsat Collection 2

### Project description

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ field measurements are available. CoastSat enables the non-expert user to extract shorelines from Landsat 5, Landsat 7, Landsat 8, Landsat 9 and Sentinel-2 images.
The shoreline detection algorithm implemented in CoastSat is optimised for sandy beach coastlines. It combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

The toolbox has four main functionalities:
1. assisted retrieval from Google Earth Engine of all available satellite images spanning the user-defined region of interest and time period.
2. automated extraction of shorelines from all the selected images using a sub-pixel resolution technique.
3. intersection of the 2D shorelines with user-defined shore-normal transects.
4. tidal correction using measured water levels and an estimate of the beach slope.
5. post-processing of the shoreline time-series, despiking and seasonal averaging.

## 1. Installation

### 1.1 Create an environment with Anaconda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will use **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have it installed on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command (change directory) to go the folder where you have downloaded this repository.

Create a new environment named `coastsat` with all the required packages by entering these commands in succession:

```
conda create -n coastsat python=3.8
conda activate coastsat
conda install -c conda-forge geopandas earthengine-api scikit-image matplotlib astropy notebook -y
pip install pyqt5
```

All the required packages have now been installed in an environment called `coastsat`. Always make sure that the environment is activated with:

```
conda activate coastsat
```

To confirm that you have successfully activated CoastSat, your terminal command line prompt should now start with (coastsat).

:warning: **In case errors are raised** :warning:: clean things up with the following command (better to have the Anaconda Prompt open as administrator) before attempting to install `coastsat` again:
```
conda clean --all
```

You can also install the packages with the **Anaconda Navigator**, in the *Environments* tab. For more details, the following [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) shows how to create and manage an environment with Anaconda.

### 1.2 Activate Google Earth Engine Python API

First, you need to request access to Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests.

Once your request has been approved, with the `coastsat` environment activated, run the following command on the Anaconda Prompt to link your environment to the GEE server:

```
earthengine authenticate
```

A web browser will open, login with a gmail account and accept the terms and conditions. Then copy the authorization code into the Anaconda terminal.

Now you are ready to start using the CoastSat toolbox!

**Note**: remember to always activate the environment with `conda activate coastsat` each time you are preparing to use the toolbox.

## 2. Usage

An example of how to run the software in a Jupyter Notebook is provided in the repository (`example_jupyter.ipynb`). To run this, first activate your `coastsat` environment with `conda activate coastsat` (if not already active), and then type:

```
jupyter notebook
```

A web browser window will open. Point to the directory where you downloaded this repository and click on `example_jupyter.ipynb`.

The following sections guide the reader through the different functionalities of CoastSat with an example at Narrabeen-Collaroy beach (Australia). If you prefer to use **Spyder**, **PyCharm** or other integrated development environments (IDEs), a Python script named `example.py` is also included in the repository.

If using `example.py` on **Spyder**, make sure that the Graphics Backend is set to **Automatic** and not **Inline** (as this mode doesn't allow to interact with the figures). To change this setting go under Preferences>IPython console>Graphics.

A Jupyter Notebook combines formatted text and code. To run the code, place your cursor inside one of the code sections and click on the `run cell` button (or press `Shift` + `Enter`) and progress forward.

![image](https://user-images.githubusercontent.com/7217258/165960239-e8870f7e-0dab-416e-bbdd-089b136b7d20.png)


### 2.1 Retrieval of the satellite images

To retrieve from the GEE server the available satellite images cropped around the user-defined region of coastline for the particular time period of interest, the following variables are required:
- `polygon`: the coordinates of the region of interest (longitude/latitude pairs in WGS84)
- `dates`: dates over which the images will be retrieved (e.g., `dates = ['2017-12-01', '2018-01-01']`)
- `sat_list`: satellite missions to consider (e.g., `sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']` for Landsat 5, 7, 8, 9 and Sentinel-2 collections)
- `sitename`: name of the site (this is the name of the subfolder where the images and other accompanying files will be stored)
- `filepath`: filepath to the directory where the data will be stored
- :new: `landsat_collection`: whether to use Collection 1 (`C01`) or Collection 2 (`C02`). Note that after 2022/01/01, Landsat images are only available in Collection 2. Landsat 9 is therefore only available as Collection 2. So if the user has selected `C01`, images prior to 2022/01/01 will be downloaded from Collection 1, while images captured after that date will be automatically taken from `C02`. Also note that at the time of writing `C02` is not complete in Google Earth Engine and still being uploaded.

The call `metadata = SDS_download.retrieve_images(inputs)` will launch the retrieval of the images and store them as .TIF files (under */filepath/sitename*). The metadata contains the exact time of acquisition (in UTC time) of each image, its projection and its geometric accuracy. If the images have already been downloaded previously and the user only wants to run the shoreline detection, the metadata can be loaded directly by running `metadata = SDS_download.get_metadata(inputs)`.

The screenshot below shows an example of inputs that will retrieve all the images of Collaroy-Narrabeen (Australia) acquired by Sentinel-2 in December 2017.

![doc1](https://user-images.githubusercontent.com/7217258/166197244-9f41de17-f387-40a6-945e-8a78b581c4b1.png)

**Note:** The area of the polygon should not exceed 100 km2, so for very long beaches split it into multiple smaller polygons.

### 2.2 Shoreline detection

To map the shorelines, the following user-defined settings are needed:
- `cloud_thresh`: threshold on maximum cloud cover that is acceptable on the images (value between 0 and 1 - this may require some initial experimentation).
- `dist_clouds`: buffer around cloud pixels where shoreline is not mapped (in metres)
- `output_epsg`: epsg code defining the spatial reference system of the shoreline coordinates. It has to be a cartesian coordinate system (i.e. projected) and not a geographical coordinate system (in latitude and longitude angles). See http://spatialreference.org/ to find the EPSG number corresponding to your local coordinate system. If unsure, use 3857 which is the web-mercator.
- `check_detection`: if set to `True` the user can quality control each shoreline detection interactively (recommended when mapping shorelines for the first time) and accept/reject each shoreline.
- `adjust_detection`: in case users wants more control over the detected shorelines, they can set this parameter to `True`, then they will be able to manually adjust the threshold used to map the shoreline on each image.
- `save_figure`: if set to `True` a figure of each mapped shoreline is saved under */filepath/sitename/jpg_files/detection*, even if the two previous parameters are set to `False`. Note that this may slow down the process.

There are additional parameters (`min_beach_size`, `buffer_size`, `min_length_sl`, `cloud_mask_issue`, `sand_color` and `pan_off` that can be tuned to optimise the shoreline detection (for Advanced users only). For the moment leave these parameters set to their default values, we will see later how they can be modified.

An example of settings is provided here:

![image](https://user-images.githubusercontent.com/7217258/182158840-ef1c527c-6ddb-44ab-a6fc-f4b46c8b0127.png)

Once all the settings have been defined, the batch shoreline detection can be launched by calling:
```
output = SDS_shoreline.extract_shorelines(metadata, settings)
```
When `check_detection` is set to `True`, a figure like the one below appears and asks the user to manually accept/reject each detection by pressing **on the keyboard** the `right arrow` (⇨) to `keep` the shoreline or `left arrow` (⇦) to `skip` the mapped shoreline. The user can break the loop at any time by pressing `escape` (nothing will be saved though).

![map_shorelines](https://user-images.githubusercontent.com/7217258/60766769-fafda480-a0f1-11e9-8f91-419d848ff98d.gif)

When `adjust_detection` is set to `True`, a figure like the one below appears and the user can adjust the position of the shoreline by clicking on the histogram of MNDWI pixel intensities. Once the threshold has been adjusted, press `Enter` and then accept/reject the image with the keyboard arrows.

![Alt text](https://github.com/kvos/CoastSat/blob/master/doc/adjust_shorelines.gif)

Once all the shorelines have been mapped, the output is available in two different formats (saved under */filepath/data/sitename*):
- `sitename_output.pkl`: contains a list with the shoreline coordinates, the exact timestamp at which the image was captured (UTC time), the geometric accuracy and the cloud cover of each individual image. This list can be manipulated with Python, a snippet of code to plot the results is provided in the example script.
- `sitename_output.geojson`: this output can be visualised in a GIS software (e.g., QGIS, ArcGIS).

The figure below shows how the satellite-derived shorelines can be opened in a GIS software (QGIS) using the `.geojson` output. Note that the coordinates in the `.geojson` file are in the spatial reference system defined by the `output_epsg`.

<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/7217258/49361401-15bd0480-f730-11e8-88a8-a127f87ca64a.jpeg">
</p>

#### Reference shoreline

Before running the batch shoreline detection, there is the option to manually digitize a reference shoreline on one cloud-free image. This reference shoreline helps to reject outliers and false detections when mapping shorelines as it only considers as valid shorelines the points that are within a defined distance from this reference shoreline. **It is highly recommended to use a reference shoreline**.

 The user can manually digitize one or several reference shorelines on one of the images by calling:
```
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_manual(metadata, settings)
settings['max_dist_ref'] = 100 # max distance (in meters) allowed from the reference shoreline
```
This function allows the user to click points along the shoreline on cloud-free satellite images, as shown in the animation below.

![ref_shoreline](https://user-images.githubusercontent.com/7217258/70408922-063c6e00-1a9e-11ea-8775-fc62e9855774.gif)

The maximum distance (in metres) allowed from the reference shoreline is defined by the parameter `max_dist_ref`. This parameter is set to a default value of 100 m. If you think that 100 m buffer from the reference shoreline will not capture the shoreline variability at your site, increase the value of this parameter. This may be the case for large nourishments or eroding/accreting coastlines.

#### Advanced shoreline detection parameters

As mentioned above, there are some additional parameters that can be modified to optimise the shoreline detection:
- `min_beach_area`: minimum allowable object area (in metres^2) for the class 'sand'. During the image classification, some features (for example, building roofs) may be incorrectly labelled as sand. To correct this, all the objects classified as sand containing less than a certain number of connected pixels are removed from the sand class. The default value is 4500 m^2, which corresponds to 20 connected pixels of 15 m^2. If you are looking at a very small beach (<20 connected pixels on the images), try decreasing the value of this parameter.
- `min_length_sl`: minimum length (in metres) of shoreline perimeter to be valid. This can be used to discard small features that are detected but do not correspond to the actual shoreline. The default value is 500 m. If the shoreline that you are trying to map is shorter than 500 m, decrease the value of this parameter.
- `cloud_mask_issue`: the cloud mask algorithm applied to Landsat images by USGS, namely CFMASK, does have difficulties sometimes with very bright features such as beaches or white-water in the ocean. This may result in pixels corresponding to a beach being identified as clouds and appear as masked pixels on your images. If this issue seems to be present in a large proportion of images from your local beach, you can switch this parameter to `True` and CoastSat will remove from the cloud mask the pixels that form very thin linear features, as often these are beaches and not clouds. Only activate this parameter if you observe this very specific cloud mask issue, otherwise leave to the default value of `False`.
- `sand_color`: this parameter can take 3 values: `default`, `latest`, `dark` or `bright`. Only change this parameter if you are seing that with the `default` the sand pixels are not being classified as sand (in orange). If your beach has dark sand (grey/black sand beaches), you can set this parameter to `dark` and the classifier will be able to pick up the dark sand. On the other hand, if your beach has white sand and the `default` classifier is not picking it up, switch this parameter to `bright`. The `latest` classifier contains all the training data and can pick up sand in most environments (but not as accurately). At this stage the different classifiers are only available for Landsat images (soon for Sentinel-2 as well).
- `pan_off`: by default Landsat 7, 8 and 9 images are pan-sharpened using the panchromatic band and a PCA algorithm. If for any reason you prefer not to pan-sharpen the Landsat images, switch it off by setting `pan_off` to `True`.
#### Re-training the classifier
CoastSat's shoreline mapping alogorithm uses an image classification scheme to label each pixel into 4 classes: sand, water, white-water and other land features. While this classifier has been trained using a wide range of different beaches, it may be that it does not perform very well at specific sites that it has never seen before. You can train a new classifier with site-specific training data in a few minutes by following the Jupyter notebook in [re-train CoastSat classifier](https://github.com/kvos/CoastSat/blob/master/doc/train_new_classifier.md).

### 2.3 Shoreline change analysis

This section shows how to obtain time-series of shoreline change along shore-normal transects. Each transect is defined by two points, its origin and a second point that defines its length and orientation. The origin is always defined first and located landwards, the second point is located seawards. There are 3 options to define the coordinates of the transects:
1. Interactively draw shore-normal transects along the mapped shorelines:
```
transects = SDS_transects.draw_transects(output, settings)
```
2. Load the transect coordinates from a .geojson file:
```
transects = SDS_tools.transects_from_geojson(path_to_geojson_file)
```
3. Create the transects by manually providing the coordinates of two points:
```
transects = dict([])
transects['Transect 1'] = np.array([[342836, ,6269215], [343315, 6269071]])
transects['Transect 2'] = np.array([[342482, 6268466], [342958, 6268310]])
transects['Transect 3'] = np.array([[342185, 6267650], [342685, 6267641]])
```

**Note:** if you choose option 2 or 3, make sure that the points that you are providing are in the spatial reference system defined by `settings['output_epsg']`.

Once the shore-normal transects have been defined, the intersection between the 2D shorelines and the transects is computed with the following function:
```
settings['along_dist'] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings)
```
The parameter `along_dist` defines the along-shore distance around the transect over which shoreline points are selected to compute the intersection. The default value is 25 m, which means that the intersection is computed as the median of the points located within 25 m of the transect (50 m alongshore-median). This helps to smooth out localised water levels in the swash zone.

An example is shown in the animation below:

![transects](https://user-images.githubusercontent.com/7217258/49990925-8b985a00-ffd3-11e8-8c54-57e4bf8082dd.gif)

There is also the option to run `SDS_transects.compute_intersection_QA()`, this function provides more quality-control when computing the intersections between shorelines and transects (small loops, multiple intersections etc).

It is recommended to use this function as it can provide cleaner shoreline time-series. See the [Jupyter Notebook](https://github.com/kvos/CoastSat/blob/master/example_jupyter.ipynb) for a detailed description of the parameters. An example of parameters for the quality control are provided below:

![image](https://user-images.githubusercontent.com/7217258/182160883-5edfb8f9-e668-440c-b55c-87e8697a2b64.png)


### 2.4 Tidal Correction

Each satellite image is captured at a different stage of the tide, therefore a tidal correction is necessary to remove the apparent shoreline changes cause by tidal fluctuations.

In order to tidally-correct the time-series of shoreline change you will need the following data:
- Time-series of water/tide level: this can be formatted as a .csv file, an example is provided [here](https://github.com/kvos/CoastSat/blob/master/examples/NARRA_tides.csv). Make sure that the dates are in UTC time as the CoastSat shorelines are always in UTC time. Also the vertical datum needs to be approx. Mean Sea Level.

- An estimate of the beach-face slope along each transect. If you don't have this data you can obtain it using [CoastSat.slope](https://github.com/kvos/CoastSat.slope), see [Vos et al. 2020](https://doi.org/10.1029/2020GL088365) for more details (preprint available [here](https://www.essoar.org/doi/10.1002/essoar.10502903.2)).

Wave setup and runup corrections are not included in the toolbox, but for more information on these additional corrections see [Castelle et al. 2021](https://doi.org/10.1016/j.geomorph.2021.107707).

### 2.5 Post-processing

The tidally-corrected time-series can be post-processed to remove outliers with a despiking algorithm in `SDS_transects.reject_outliers()`.

![image](https://user-images.githubusercontent.com/7217258/182162154-9d8da81d-a5fc-486e-baf6-55e2a5782096.png)

Functions to compute seasonal and monthly averages on the shoreline time-series are also provided: `SDS_transects.seasonal_averages()` and `SDS_transects.monthly_averages()`.

![NA1](https://user-images.githubusercontent.com/7217258/182162937-58bad8f1-35c7-4789-a03c-05799380bacf.jpg)


## Issues
Having a problem? Post an issue in the [Issues page](https://github.com/kvos/coastsat/issues) (please do not email).

## Contributing
If you are willing to contribute, check out our todo list in the [Projects page](https://github.com/kvos/CoastSat/projects/1).
1. Fork the repository (https://github.com/kvos/coastsat/fork).
A fork is a copy on which you can make your changes.
2. Create a new branch on your fork
3. Commit your changes and push them to your branch
4. When the branch is ready to be merged, create a Pull Request (how to make a clean pull request explained [here](https://gist.github.com/MarcDiethelm/7303312))

## References

1. Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. *Environmental Modelling and Software*. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)

2. Vos K., Harley M.D., Splinter K.D., Simmons J.A., Turner I.L. (2019). Sub-annual to multi-decadal shoreline variability from publicly available satellite imagery. *Coastal Engineering*. 150, 160–174. https://doi.org/10.1016/j.coastaleng.2019.04.004

3. Vos K., Harley M.D., Splinter K.D., Walker A., Turner I.L. (2020). Beach slopes from satellite-derived shorelines. *Geophysical Research Letters*. 47(14). https://doi.org/10.1029/2020GL088365 (Open Access preprint [here](https://www.essoar.org/doi/10.1002/essoar.10502903.2))

4. Castelle B., Masselink G., Scott T., Stokes C., Konstantinou A., Marieu V., Bujan S. (2021). Satellite-derived shoreline detection at a high-energy meso-macrotidal beach. *Geomorphology*. volume 383, 107707. https://doi.org/10.1016/j.geomorph.2021.107707

5. Vos, K. and Deng, W. and Harley, M. D. and Turner, I. L. and Splinter, K. D. M. (2022). Beach-face slope dataset for Australia. *Earth System Science Data*. volume 14, 3, p. 1345--1357. https://doi.org/10.5194/essd-14-1345-2022

6. Training dataset used for pixel-wise classification in CoastSat: https://doi.org/10.5281/zenodo.3334147
