#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# Kilian Vos WRL 2018

#%% 1. Initial settings

# load modules
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.ion()
import pandas as pd
from datetime import datetime
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

# region of interest (longitude, latitude in WGS84)
polygon = [[[151.301454, -33.700754],
            [151.311453, -33.702075],
            [151.307237, -33.739761],
            [151.294220, -33.736329],
            [151.301454, -33.700754]]]
# can also be loaded from a .kml polygon
#kml_polygon = os.path.join(os.getcwd(), 'examples', 'NARRA_polygon.kml')
#polygon = SDS_tools.polygon_from_kml(kml_polygon)
       
# date range
dates = ['2017-12-01', '2018-01-01']

# satellite missions
sat_list = ['S2']

# name of the site
sitename = 'NARRA'

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), 'data')

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data
        }

# before downloading the images, check how many images are available for your inputs
SDS_download.check_images_available(inputs);

#%% 2. Retrieve images

# only uncomment this line if you want Landsat Tier 2 images (not suitable for time-series analysis)
# inputs['include_T2'] = True

# retrieve satellite images from GEE
metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs) 

#%% 3. Batch shoreline detection
    
# settings for the shoreline extraction
settings = { 
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'output_epsg': 3857,        # epsg code of spatial reference system desired for the output   
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection to the user for validation
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # add the inputs defined previously
    'inputs': inputs,
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'sand_color': 'default',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
}

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100        

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

# remove duplicates (images taken on the same date by the same satellite)
output = SDS_tools.remove_duplicates(output)
# remove inaccurate georeferencing (set threshold to 10 m)
output = SDS_tools.remove_inaccurate_georef(output, 10)

# for GIS applications, save output into a GEOJSON layer
geomtype = 'lines' # choose 'points' or 'lines' for the layer geometry
gdf = SDS_tools.output_to_gdf(output, geomtype)
gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
# save GEOJSON layer to file
gdf.to_file(os.path.join(inputs['filepath'], inputs['sitename'], '%s_output_%s.geojson'%(sitename,geomtype)),
                                driver='GeoJSON', encoding='utf-8')

# plot the mapped shorelines
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend()    

#%% 4. Shoreline analysis

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f) 

# now we have to define cross-shore transects over which to quantify the shoreline changes
# each transect is defined by two points, its origin and a second point that defines its orientation

# there are 3 options to create the transects:
# - option 1: draw the shore-normal transects along the beach
# - option 2: load the transect coordinates from a .kml file
# - option 3: create the transects manually by providing the coordinates

# option 1: draw origin of transect first and then a second point to define the orientation
# transects = SDS_transects.draw_transects(output, settings)
    
# option 2: load the transects from a .geojson file
geojson_file = os.path.join(os.getcwd(), 'examples', 'NARRA_transects.geojson')
transects = SDS_tools.transects_from_geojson(geojson_file)

# option 3: create the transects by manually providing the coordinates of two points 
# transects = dict([])
# transects['NA1'] = np.array([[16843142, -3989358], [16843457, -3989535]])
# transects['NA2'] = np.array([[16842958, -3989834], [16843286, -3989983]])
# transects['NA3'] = np.array([[16842602, -3990878], [16842955, -3990949]])
# transects['NA4'] = np.array([[16842596, -3991929], [16842955, -3991895]])
# transects['NA5'] = np.array([[16842838, -3992900], [16843155, -3992727]])
   
# plot the transects to make sure they are correct (origin landwards!)
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
for i,key in enumerate(list(transects.keys())):
    plt.plot(transects[key][0,0],transects[key][0,1], 'bo', ms=5)
    plt.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
    plt.text(transects[key][0,0]-100, transects[key][0,1]+100, key,
                va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))

# intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
# (also saved a .csv file with the time-series, dates are in UTC time)
settings['along_dist'] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings) 

# plot the time-series
fig = plt.figure(figsize=[15,8], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-50,50])
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w')
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14) 
    
#%% 4. Tidal correction
    
# For this example, measured water levels for Sydney are stored in a csv file located in /examples.
# When using your own file make sure that the dates are in UTC time, as the CoastSat shorelines are also in UTC
# and the datum for the water levels is approx. Mean Sea Level. We assume a beach slope of 0.1 here.

# load the measured tide data
filepath = os.path.join(os.getcwd(),'examples','NARRA_tides.csv')
tide_data = pd.read_csv(filepath, parse_dates=['dates'])
dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
tides_ts = np.array(tide_data['tide'])

# get tide levels corresponding to the time of image acquisition
dates_sat = output['dates']
tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)

# plot the subsampled tide data
fig, ax = plt.subplots(1,1,figsize=(15,4), tight_layout=True)
ax.grid(which='major', linestyle=':', color='0.5')
ax.plot(tide_data['dates'], tide_data['tide'], '-', color='0.6', label='all time-series')
ax.plot(dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w',lw=1, label='image acquisition')
ax.set(ylabel='tide level [m]',xlim=[dates_sat[0],dates_sat[-1]], title='Water levels at the time of image acquisition')
ax.legend()

# tidal correction along each transect
reference_elevation = 0 # elevation at which you would like the shoreline time-series to be
beach_slope = 0.1
cross_distance_tidally_corrected = {}
for key in cross_distance.keys():
    correction = (tides_sat-reference_elevation)/beach_slope
    cross_distance_tidally_corrected[key] = cross_distance[key] + correction
    
# store the tidally-corrected time-series in a .csv file
out_dict = dict([])
out_dict['dates'] = dates_sat
for key in cross_distance_tidally_corrected.keys():
    out_dict['Transect '+ key] = cross_distance_tidally_corrected[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                  'transect_time_series_tidally_corrected.csv')
df.to_csv(fn, sep=',')
print('Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s'%fn)

# plot the time-series of shoreline change (both raw and tidally-corrected)
fig = plt.figure(figsize=[15,8], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-50,50])
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w', label='raw')
    ax.plot(output['dates'], cross_distance_tidally_corrected[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w', label='tidally-corrected')
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)
ax.legend()