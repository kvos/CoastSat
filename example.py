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
from scipy import interpolate
from scipy import stats
from datetime import datetime, timedelta
import pytz
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
import cv2
import imageio
from PIL import Image
import tkinter as tk

# region of interest (longitude, latitude in WGS84)
polygon = [[[151.301454, -33.700754],
            [151.311453, -33.702075],
            [151.307237, -33.739761],
            [151.294220, -33.736329],
            [151.301454, -33.700754]]]
# can also be loaded from a .kml polygon
# kml_polygon = os.path.join(os.getcwd(), 'examples', 'NARRA_polygon.kml')
# polygon = SDS_tools.polygon_from_kml(kml_polygon)
# or read from geojson polygon (create it from https://geojson.io/)
# geojson_polygon = os.path.join(os.getcwd(), 'examples', 'NARRA_polygon.geojson')
# polygon = SDS_tools.polygon_from_geojson(geojson_polygon)
# convert polygon to a smallest rectangle (sides parallel to coordinate axes)
polygon = SDS_tools.smallest_rectangle(polygon)

# date range
dates = ['1984-01-01', '2022-01-01']

# satellite missions
sat_list = ['L5','L7','L8']
collection = 'C02' # choose Landsat collection 'C01' or 'C02'
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
    'filepath': filepath_data,
    'landsat_collection': collection
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
    'cloud_thresh': 0.1,        # threshold on maximum cloud cover
    'dist_clouds': 300,         # ditance around clouds where shoreline can't be mapped
    'output_epsg': 28356,       # epsg code of spatial reference system desired for the output
    # quality control:
    'check_detection': False,    # if True, shows each shoreline detection to the user for validation
    'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 1000,     # minimum area (in metres^2) for an object to be labelled as a beach
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images
    'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    'pan_off': False,           # True to switch pansharpening off for Landsat 7/8/9 imagery
    's2cloudless_prob': 40,      # threshold to identify cloud pixels in the s2cloudless probability mask
    # add the inputs defined previously
    'inputs': inputs,
}

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
#SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)

# Function to check if jpg files already exist
def check_files_exist(path, file_extension=".jpg"):
    if not os.path.exists(path):
        return False
    return any(file.endswith(file_extension) for file in os.listdir(path))

# Define the path to preprocessed jpg files in the current directory
preprocessed_path = f"./data/{sitename}/jpg_files/preprocessed"

# Check if directory exists and if files exist
if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
    print(f"Directory created: {preprocessed_path}")

if not check_files_exist(preprocessed_path):
    # Only save images if they don't already exist
    SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)
    print("Satellite images saved as .jpg in", preprocessed_path)
else:
    print("Preprocessed jpg files already exist in", preprocessed_path)

# Define functions
def resize_image_to_macro_block(image, macro_block_size=16):
    height, width = image.shape[:2]
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    return cv2.resize(image, (new_width, new_height))

def load_and_resize_images(image_folder, macro_block_size=16):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(image_folder, filename)
            image = np.array(Image.open(img_path))
            resized_image = resize_image_to_macro_block(image, macro_block_size)
            images.append(resized_image)
    return images

def create_animation(image_folder, output_file, fps=4, macro_block_size=16, probesize=5000000):
    images = load_and_resize_images(image_folder, macro_block_size)
    writer = imageio.get_writer(output_file, fps=fps, ffmpeg_params=['-probesize', str(probesize)])
    for image in images:
        writer.append_data(image)
    writer.close()

#create MP4 timelapse animation
#fn_animation = os.path.join(inputs['filepath'],inputs['sitename'], '%s_animation_RGB.mp4'%inputs['sitename'])
#fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'preprocessed')
#fps = 4 # frames per second in animation
#SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)

#IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (2700, 1350) to (2704, 1360) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
#[rawvideo @ 0x5b94500] Stream #0: not enough frames to estimate rate; consider increasing probesize
#[swscaler @ 0x5babfc0] Warning: data is not aligned! This can lead to a speed loss

# Define paths and settings
print("create MP4 timelapse animation")
fn_animation = os.path.join(inputs['filepath'], inputs['sitename'], '%s_animation_RGB.mp4' % inputs['sitename'])
fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'preprocessed')
fps = 4
create_animation(fp_images, fn_animation, fps)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
#settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
#AttributeError: '_tkinter.tkapp' object has no attribute 'showMaximized'

# Replace showMaximized with state('zoomed')
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
geomtype = 'points' # choose 'points' or 'lines' for the layer geometry
gdf = SDS_tools.output_to_gdf(output, geomtype)
if gdf is None:
    raise Exception("output does not contain any mapped shorelines")
gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
# save GEOJSON layer to file
gdf.to_file(os.path.join(inputs['filepath'], inputs['sitename'], '%s_output_%s.geojson'%(sitename,geomtype)),
                                driver='GeoJSON', encoding='utf-8')

# create MP4 timelapse animation
print("create MP4 timelapse animation")
fn_animation = os.path.join(inputs['filepath'],inputs['sitename'], '%s_animation_shorelines.mp4'%inputs['sitename'])
fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'detection')
fps = 4 # frames per second in animation
create_animation(fp_images, fn_animation, fps)

# plot the mapped shorelines
plt.ion()
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
# plt.legend()
fig.savefig(os.path.join(inputs['filepath'], inputs['sitename'], 'mapped_shorelines.jpg'),dpi=200)

#%% 4. Shoreline analysis

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f)
# remove duplicates (images taken on the same date by the same satellite)
output = SDS_tools.remove_duplicates(output)
# remove inaccurate georeferencing (set threshold to 10 m)
output = SDS_tools.remove_inaccurate_georef(output, 10)

# now we have to define cross-shore transects over which to quantify the shoreline changes
# each transect is defined by two points, its origin and a second point that defines its orientation

# there are 3 options to create the transects:
# - option 1: draw the shore-normal transects along the beach
# - option 2: load the transect coordinates from a .kml file
# - option 3: create the transects manually by providing the coordinates

# option 1: draw origin of transect first and then a second point to define the orientation
# transects = SDS_transects.draw_transects(output, settings)

# option 2: load the transects from a .geojson file

# Navigate to the 'examples' directory and load the geojson file
current_dir = os.path.abspath(os.path.dirname(__file__))
geojson_file = os.path.abspath(os.path.join(current_dir, 'examples', 'NARRA_transects.geojson'))
transects = SDS_tools.transects_from_geojson(geojson_file)

# Verify the path to ensure it is correct
print(f"The geojson file path is: {geojson_file}")

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
fig.savefig(os.path.join(inputs['filepath'], inputs['sitename'], 'mapped_shorelines.jpg'),dpi=200)

#%% Option 1: Compute intersections with quality-control parameters (recommended)

settings_transects = { # parameters for computing intersections
                      'along_dist':          25,        # along-shore distance to use for computing the intersection
                      'min_points':          3,         # minimum number of shoreline points to calculate an intersection
                      'max_std':             15,        # max std for points around transect
                      'max_range':           30,        # max range for points around transect
                      'min_chainage':        -100,      # largest negative value along transect (landwards of transect origin)
                      'multiple_inter':      'auto',    # mode for removing outliers ('auto', 'nan', 'max')
                      'auto_prc':            0.1,      # percentage to use in 'auto' mode to switch from 'nan' to 'max'
                     }
cross_distance = SDS_transects.compute_intersection_QC(output, transects, settings_transects)

#%% Option 2: Conpute intersection in a simple way (no quality-control)

# settings['along_dist'] = 25
# cross_distance = SDS_transects.compute_intersection(output, transects, settings)

#%% Plot the time-series of cross-shore shoreline change

fig = plt.figure(figsize=[15,8], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-50,50])
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=4, mfc='w')
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)
fig.savefig(os.path.join(inputs['filepath'], inputs['sitename'], 'time_series_raw.jpg'),dpi=200)

# save time-series in a .csv file
out_dict = dict([])
out_dict['dates'] = output['dates']
for key in transects.keys():
    out_dict['Transect '+ key] = cross_distance[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],
                  'transect_time_series.csv')
df.to_csv(fn, sep=',')
print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)

#%% 4. Tidal correction

# For this example, measured water levels for Sydney are stored in a csv file located in /examples.
# When using your own file make sure that the dates are in UTC time, as the CoastSat shorelines are also in UTC
# and the datum for the water levels is approx. Mean Sea Level. We assume a beach slope of 0.1 here.

# Load the measured tide data
filepath = os.path.abspath(os.path.join(current_dir, 'examples', 'NARRA_tides.csv'))

# Verify the path to ensure it is correct
print(f"The file path is: {filepath}")

tide_data = pd.read_csv(filepath, parse_dates=['dates'])
dates_ts = [pd.to_datetime(_).to_pydatetime() for _ in tide_data['dates']]
tides_ts = np.array(tide_data['tide'])

# Print the dates of the time series
print("Dates of the time series (dates_ts):")
print(dates_ts)

# Check if dates_ts is not empty
if not dates_ts:
    raise ValueError("The time series dates (dates_ts) are empty.")

time_series_start = min(dates_ts)
time_series_end = max(dates_ts)

# Assuming 'output' is defined somewhere in your script
if 'output' in globals() and 'dates' in output:
    dates_sat = output['dates']
else:
    raise ValueError("The 'output' variable or 'output['dates']' is not defined.")

# Print the dates of the input data
print("Dates of the input data (dates_sat):")
print(dates_sat)

# Check if dates_sat is not empty
if not dates_sat:
    raise ValueError("The input dates (dates_sat) are empty.")

input_data_start = min(dates_sat)
input_data_end = max(dates_sat)

# Print the date ranges
print(f"Date range of the time series: {time_series_start} to {time_series_end}")
print(f"Date range of the input data: {input_data_start} to {input_data_end}")

# Check if the time series covers the input data range
if input_data_start < time_series_start or input_data_end > time_series_end:
    print("Sorry, the time series data does not cover the range of your input dates.")
    print(f"The available time series data ranges from {time_series_start} to {time_series_end}.")
    
    # Adjust the dates to fit within the available time series data range
    adjusted_dates_sat = [
        max(time_series_start, min(time_series_end, date)) for date in dates_sat
    ]
    
    # Ensure adjusted_dates_sat is not empty
    if not adjusted_dates_sat:
        raise ValueError("The adjusted input dates are empty after adjustment.")
    
    print("Adjusting input dates to fit within the available time series data range:")
    print(f"Adjusted date range: {min(adjusted_dates_sat)} to {max(adjusted_dates_sat)}")
else:
    adjusted_dates_sat = dates_sat

# Now proceed with your processing using adjusted_dates_sat
try:
    if len(adjusted_dates_sat) > 0:
        tides_sat = SDS_tools.get_closest_datapoint(adjusted_dates_sat, dates_ts, tides_ts)
        print("Tide levels corresponding to the input dates have been successfully retrieved.")
    else:
        tides_sat = np.array([])
        print("Adjusted dates are empty, skipping tide level retrieval.")
except Exception as e:
    print(f"An error occurred: {e}")
    tides_sat = np.array([])

# Plotting the subsampled tide data
fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
ax.grid(which='major', linestyle=':', color='0.5')
ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
if tides_sat.size > 0:
    ax.plot(adjusted_dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w', lw=1, label='image acquisition')
    ax.set(ylabel='tide level [m]', xlim=[adjusted_dates_sat[0], adjusted_dates_sat[-1]], title='Water levels at the time of image acquisition')
else:
    print("Tide levels for the input dates could not be plotted because tides_sat is empty.")
ax.legend()
plt.show()

# Further processing only if tides_sat is not empty
if tides_sat.size > 0:
    try:
        reference_elevation = 0  # Example reference elevation
        beach_slope = 0.1  # Example beach slope
        correction = (tides_sat - reference_elevation) / beach_slope
        print("Correction values computed successfully.")
    except NameError as e:
        print(f"A NameError occurred: {e}. Did you mean: 'tide_data'?")
    except Exception as e:
        print(f"An error occurred while computing correction values: {e}")
else:
    print("Skipping correction computation because tides_sat is empty.")

# Tidal correction along each transect
reference_elevation = 0.7  # Example reference elevation
beach_slope = 0.1  # Example beach slope
cross_distance_tidally_corrected = {}
if tides_sat.size > 0:
    for key in cross_distance.keys():
        correction = (tides_sat - reference_elevation) / beach_slope
        cross_distance_tidally_corrected[key] = cross_distance[key] + correction
else:
    print("Skipping tidal correction along transects because tides_sat is empty.")

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

#%% 5. Time-series post-processing

# load mapped shorelines from 1984 (mapped with the previous code)
filename_output = os.path.abspath(os.path.join(current_dir, 'examples', 'NARRA_output.pkl'))

with open(filename_output, 'rb') as f:
    output = pickle.load(f)

# plot the mapped shorelines
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
plt.title('%d shorelines mapped at Narrabeen from 1984'%len(output['shorelines']))
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
for i,key in enumerate(list(transects.keys())):
    plt.plot(transects[key][0,0],transects[key][0,1], 'bo', ms=5)
    plt.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
    plt.text(transects[key][0,0]-100, transects[key][0,1]+100, key,
                va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))

# load long time-series (1984-2021)
filepath = os.path.abspath(os.path.join(current_dir, 'examples', 'NARRA_time_series_tidally_corrected.csv'))

df = pd.read_csv(filepath, parse_dates=['dates'])
dates = [_.to_pydatetime() for _ in df['dates']]
cross_distance = dict([])
for key in transects.keys():
    cross_distance[key] = np.array(df[key])

#%% 5.1 Remove outliers

# plot Otsu thresholds for the mapped shorelines
fig,ax = plt.subplots(1,1,figsize=[12,5],tight_layout=True)
ax.grid(which='major',ls=':',lw=0.5,c='0.5')
ax.plot(output['dates'],output['MNDWI_threshold'],'o-',mfc='w')
ax.axhline(y=-0.5,ls='--',c='r',label='otsu_threshold limits')
ax.axhline(y=0,ls='--',c='r')
ax.set(title='Otsu thresholds on MNDWI for the %d shorelines mapped'%len(output['shorelines']),
       ylim=[-0.6,0.2],ylabel='otsu threshold')
ax.legend(loc='upper left')
fig.savefig(os.path.join(inputs['filepath'], inputs['sitename'], 'otsu_threhsolds.jpg'))

# remove outliers in the time-series (despiking)
settings_outliers = {'otsu_threshold':     [-.5,0],        # min and max intensity threshold use for contouring the shoreline
                     'max_cross_change':   40,             # maximum cross-shore change observable between consecutive timesteps
                     'plot_fig':           True,           # whether to plot the intermediate steps
                    }
cross_distance = SDS_transects.reject_outliers(cross_distance,output,settings_outliers)

#%% 5.2 Seasonal averaging

# compute seasonal averages along each transect
season_colors = {'DJF':'C3', 'MAM':'C1', 'JJA':'C2', 'SON':'C0'}
for key in cross_distance.keys():
    chainage = cross_distance[key]
    # remove nans
    idx_nan = np.isnan(chainage)
    dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
    chainage = chainage[~idx_nan]

    # compute shoreline seasonal averages (DJF, MAM, JJA, SON)
    dict_seas, dates_seas, chainage_seas, list_seas = SDS_transects.seasonal_average(dates_nonan, chainage)

    # plot seasonal averages
    fig,ax=plt.subplots(1,1,figsize=[14,4],tight_layout=True)
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.set_title('Time-series at %s'%key, x=0, ha='left')
    ax.set(ylabel='distance [m]')
    ax.plot(dates_nonan, chainage,'+', lw=1, color='k', mfc='w', ms=4, alpha=0.5,label='raw datapoints')
    ax.plot(dates_seas, chainage_seas, '-', lw=1, color='k', mfc='w', ms=4, label='seasonally-averaged')
    for k,seas in enumerate(dict_seas.keys()):
        ax.plot(dict_seas[seas]['dates'], dict_seas[seas]['chainages'],
                 'o', mec='k', color=season_colors[seas], label=seas,ms=5)
    ax.legend(loc='lower left',ncol=6,markerscale=1.5,frameon=True,edgecolor='k',columnspacing=1)

#%% 5.3 Monthly averaging

# compute monthly averages along each transect
month_colors = plt.cm.get_cmap('tab20')
for key in cross_distance.keys():
    chainage = cross_distance[key]
    # remove nans
    idx_nan = np.isnan(chainage)
    dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
    chainage = chainage[~idx_nan]

    # compute shoreline seasonal averages (DJF, MAM, JJA, SON)
    dict_month, dates_month, chainage_month, list_month = SDS_transects.monthly_average(dates_nonan, chainage)

    # plot seasonal averages
    fig,ax=plt.subplots(1,1,figsize=[14,4],tight_layout=True)
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.set_title('Time-series at %s'%key, x=0, ha='left')
    ax.set(ylabel='distance [m]')
    ax.plot(dates_nonan, chainage,'+', lw=1, color='k', mfc='w', ms=4, alpha=0.5,label='raw datapoints')
    ax.plot(dates_month, chainage_month, '-', lw=1, color='k', mfc='w', ms=4, label='monthly-averaged')
    for k,month in enumerate(dict_month.keys()):
        ax.plot(dict_month[month]['dates'], dict_month[month]['chainages'],
                 'o', mec='k', color=month_colors(k), label=month,ms=5)
    ax.legend(loc='lower left',ncol=7,markerscale=1.5,frameon=True,edgecolor='k',columnspacing=1)

#%% 6. Validation against survey data
# In this section we provide a comparison against in situ data at Narrabeen.
# See the Jupyter Notebook for information on hopw to downlaod the Narrabeen data from http://narrabeen.wrl.unsw.edu.au/

# 6.1. Read and preprocess downloaded csv file Narrabeen_Profiles.csv
# read the csv file
#df = pd.read_csv(fp_datasets)
#if Narrabeen_Profiles.csv file don't exist,   File "/usr/lib/python3/dist-packages/pandas/io/common.py", line 863, in get_handle handle = open(
#FileNotFoundError: [Errno 2] No such file or directory: 'coastsat/CoastSat-master/examples/Narrabeen_Profiles.csv'

# Define the path to the CSV file
fp_datasets = os.path.abspath(os.path.join(current_dir, 'examples', 'Narrabeen_Profiles.csv'))

# Try to read the CSV file and handle any errors that occur
try:
    df = pd.read_csv(fp_datasets)
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{fp_datasets}' does not exist. Please ensure the file path is correct and the file is present.")
    print("You can download the Narrabeen data from http://narrabeen.wrl.unsw.edu.au/")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{fp_datasets}' is empty. Please provide a valid CSV file.")
except pd.errors.ParserError:
    print(f"Error: There was an error parsing the file '{fp_datasets}'. Please ensure the file is a valid CSV.")
except Exception as e:
    print(f"An unexpected error occurred while trying to read the file '{fp_datasets}': {e}")
else:
    # If the CSV was loaded successfully, proceed with further processing
    pf_names = list(np.unique(df['Profile ID']))
    print(f"Profile IDs loaded: {pf_names}")

# select contour level
contour_level = 0.7

# initialise topo_profiles structure
topo_profiles = dict([])
for i in range(len(pf_names)):
     # read dates
    df_pf = df.loc[df['Profile ID'] == pf_names[i]]
    dates_str = df['Date']
    dates_unique = np.unique(dates_str)
    # loop through dates
    topo_profiles[pf_names[i]] = {'dates':[],'chainages':[]}
    for date in dates_unique:
        # extract chainage and elevation for that date
        df_date = df_pf.loc[dates_str == date]
        chainages = np.array(df_date['Chainage'])
        elevations = np.array(df_date['Elevation'])
        if len(chainages) == 0: continue
        # use interpolation to extract the chainage at the contour level
        f = interpolate.interp1d(elevations, chainages, bounds_error=False)
        chainage_contour_level = f(contour_level)
        topo_profiles[pf_names[i]]['chainages'].append(chainage_contour_level)
        date_utc = pytz.utc.localize(datetime.strptime(date,'%Y-%m-%d'))
        topo_profiles[pf_names[i]]['dates'].append(date_utc)

# plot time-series
fig = plt.figure(figsize=[15,8], tight_layout=True)
gs = gridspec.GridSpec(len(topo_profiles),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(topo_profiles.keys()):
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.plot(topo_profiles[key]['dates'], topo_profiles[key]['chainages'], '-o', ms=4, mfc='w')
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)
# save a .pkl file
with open(os.path.join(current_dir, 'examples', 'Narrabeen_ts_07m.pkl'), 'wb') as f:
    pickle.dump(topo_profiles, f)

#%% 6.2. Compare time-series along each transect
# load survey data
with open(os.path.join(current_dir, 'examples', 'Narrabeen_ts_07m.pkl'), 'rb') as f:
    gt = pickle.load(f)
# change names to mach surveys
for i,key in enumerate(list(cross_distance.keys())):
    key_gt = list(gt.keys())[i]
    cross_distance[key_gt] = cross_distance.pop(key)

# set parameters for comparing the two time-series
sett = {'min_days':3,   # numbers of days difference under which to use nearest neighbour interpolation
        'max_days':10,  # maximum number of days difference to do a comparison
        'binwidth':3,   # binwidth for histogram plotting
        'lims':[-50,50] # cross-shore change limits for plotting purposes
       }

# initialise variables
chain_sat_all = []
chain_sur_all = []
satnames_all = []
for key in cross_distance.keys():

    # remove nans
    chainage = cross_distance[key]
    idx_nan = np.isnan(chainage)
    dates_nonans = [output['dates'][k] for k in np.where(~idx_nan)[0]]
    satnames_nonans = [output['satname'][k] for k in np.where(~idx_nan)[0]]
    chain_nonans = chainage[~idx_nan]

    chain_sat_dm = chain_nonans
    chain_sur_dm = gt[key]['chainages']

    # plot the time-series
    fig= plt.figure(figsize=[15,8], tight_layout=True)
    gs = gridspec.GridSpec(2,3)
    ax0 = fig.add_subplot(gs[0,:])
    ax0.grid(which='major',linestyle=':',color='0.5')
    ax0.plot(gt[key]['dates'], chain_sur_dm, '-',mfc='w',ms=5,label='in situ')
    ax0.plot(dates_nonans, chain_sat_dm,'-',mfc='w',ms=5,label='satellite')
    ax0.set(title= 'Transect ' + key, xlim=[output['dates'][0]-timedelta(days=30),
                                           output['dates'][-1]+timedelta(days=30)])#,ylim=sett['lims'])
    ax0.legend()

    # interpolate surveyed data around satellite data
    chain_int = np.nan*np.ones(len(dates_nonans))
    for k,date in enumerate(dates_nonans):
        # compute the days distance for each satellite date
        days_diff = np.array([ (_ - date).days for _ in gt[key]['dates']])
        # if nothing within 10 days put a nan
        if np.min(np.abs(days_diff)) > sett['max_days']:
            chain_int[k] = np.nan
        else:
            # if a point within 3 days, take that point (no interpolation)
            if np.min(np.abs(days_diff)) < sett['min_days']:
                idx_closest = np.where(np.abs(days_diff) == np.min(np.abs(days_diff)))
                chain_int[k] = float(gt[key]['chainages'][idx_closest[0][0]])
            else: # otherwise, between 3 and 10 days, interpolate between the 2 closest points
                if sum(days_diff > 0) == 0:
                    break
                idx_after = np.where(days_diff > 0)[0][0]
                idx_before = idx_after - 1
                x = [gt[key]['dates'][idx_before].toordinal() , gt[key]['dates'][idx_after].toordinal()]
                y = [gt[key]['chainages'][idx_before], gt[key]['chainages'][idx_after]]
                f = interpolate.interp1d(x, y,bounds_error=True)
                chain_int[k] = float(f(date.toordinal()))

    # remove nans again
    idx_nan = np.isnan(chain_int)
    chain_sat = chain_nonans[~idx_nan]
    chain_sur = chain_int[~idx_nan]
    dates_sat = [dates_nonans[k] for k in np.where(~idx_nan)[0]]
    satnames = [satnames_nonans[k] for k in np.where(~idx_nan)[0]]
    chain_sat_all = np.append(chain_sat_all,chain_sat)
    chain_sur_all = np.append(chain_sur_all,chain_sur)
    satnames_all = satnames_all + satnames

    # error statistics
    slope, intercept, rvalue, pvalue, std_err = stats.linregress(chain_sur, chain_sat)
    R2 = rvalue**2
    ax0.text(0,1,'R2 = %.2f'%R2,bbox=dict(boxstyle='square', facecolor='w', alpha=1),transform=ax0.transAxes)
    chain_error = chain_sat - chain_sur
    rmse = np.sqrt(np.mean((chain_error)**2))
    mean = np.mean(chain_error)
    std = np.std(chain_error)
    q90 = np.percentile(np.abs(chain_error), 90)

    # 1:1 plot
    ax1 = fig.add_subplot(gs[1,0])
    ax1.axis('equal')
    ax1.grid(which='major',linestyle=':',color='0.5')
    for k,sat in enumerate(list(np.unique(satnames))):
        idx = np.where([_ == sat for _ in satnames])[0]
        ax1.plot(chain_sur[idx], chain_sat[idx], 'o', ms=4, mfc='C'+str(k),mec='C'+str(k), alpha=0.7, label=sat)
    ax1.legend(loc=4)
    ax1.plot([ax1.get_xlim()[0], ax1.get_ylim()[1]],[ax1.get_xlim()[0], ax1.get_ylim()[1]],'k--',lw=2)
    ax1.set(xlabel='survey [m]', ylabel='satellite [m]')

    # boxplots
    ax2 = fig.add_subplot(gs[1,1])
    data = []
    median_data = []
    n_data = []
    ax2.yaxis.grid()
    for k,sat in enumerate(list(np.unique(satnames))):
        idx = np.where([_ == sat for _ in satnames])[0]
        data.append(chain_error[idx])
        median_data.append(np.median(chain_error[idx]))
        n_data.append(len(chain_error[idx]))
    bp = ax2.boxplot(data,0,'k.', labels=list(np.unique(satnames)), patch_artist=True)
    for median in bp['medians']:
        median.set(color='k', linewidth=1.5)
    for j,boxes in enumerate(bp['boxes']):
        boxes.set(facecolor='C'+str(j))
        ax2.text(j+1,median_data[j]+1, '%.1f' % median_data[j], horizontalalignment='center', fontsize=12)
        ax2.text(j+1+0.35,median_data[j]+1, ('n=%.d' % int(n_data[j])), ha='center', va='center', fontsize=12, rotation='vertical')
    ax2.set(ylabel='error [m]', ylim=sett['lims'])

    # histogram
    ax3 = fig.add_subplot(gs[1,2])
    ax3.grid(which='major',linestyle=':',color='0.5')
    ax3.axvline(x=0, ls='--', lw=1.5, color='k')
    binwidth=sett['binwidth']
    bins = np.arange(min(chain_error), max(chain_error) + binwidth, binwidth)
    density = plt.hist(chain_error, bins=bins, density=True, color='0.6', edgecolor='k', alpha=0.5)
    mu, std = stats.norm.fit(chain_error)
    pval = stats.normaltest(chain_error)[1]
    xlims = ax3.get_xlim()
    x = np.linspace(xlims[0], xlims[1], 100)
    p = stats.norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r-', linewidth=1)
    ax3.set(xlabel='error [m]', ylabel='pdf', xlim=sett['lims'])
    str_stats = ' rmse = %.1f\n mean = %.1f\n std = %.1f\n q90 = %.1f' % (rmse, mean, std, q90)
    ax3.text(0, 0.98, str_stats,va='top', transform=ax3.transAxes)

    # save plot
    fig.savefig( os.path.join(current_dir, 'examples','comparison_transect_%s.jpg'%key), dpi=150)

#%% 6.3. Comparison for all transects

# calculate statistics for all transects together
chain_error = chain_sat_all - chain_sur_all
slope, intercept, rvalue, pvalue, std_err = stats.linregress(chain_sur, chain_sat)
R2 = rvalue**2
rmse = np.sqrt(np.mean((chain_error)**2))
mean = np.mean(chain_error)
std = np.std(chain_error)
q90 = np.percentile(np.abs(chain_error), 90)

fig,ax = plt.subplots(1,2,figsize=(15,5), tight_layout=True)
# histogram
ax[0].grid(which='major',linestyle=':',color='0.5')
ax[0].axvline(x=0, ls='--', lw=1.5, color='k')
binwidth=sett['binwidth']
bins = np.arange(min(chain_error), max(chain_error) + binwidth, binwidth)
density = ax[0].hist(chain_error, bins=bins, density=True, color='0.6', edgecolor='k', alpha=0.5)
mu, std = stats.norm.fit(chain_error)
pval = stats.normaltest(chain_error)[1]
xlims = ax3.get_xlim()
x = np.linspace(xlims[0], xlims[1], 100)
p = stats.norm.pdf(x, mu, std)
ax[0].plot(x, p, 'r-', linewidth=1)
ax[0].set(xlabel='error [m]', ylabel='pdf', xlim=sett['lims'])
str_stats = ' rmse = %.1f\n mean = %.1f\n std = %.1f\n q90 = %.1f' % (rmse, mean, std, q90)
ax[0].text(0, 0.98, str_stats,va='top', transform=ax[0].transAxes,fontsize=14)

# boxplot
data = []
median_data = []
n_data = []
ax[1].yaxis.grid()
for k,sat in enumerate(list(np.unique(satnames_all))):
    idx = np.where([_ == sat for _ in satnames_all])[0]
    data.append(chain_error[idx])
    median_data.append(np.median(chain_error[idx]))
    n_data.append(len(chain_error[idx]))
bp = ax[1].boxplot(data,0,'k.', labels=list(np.unique(satnames_all)), patch_artist=True)
for median in bp['medians']:
    median.set(color='k', linewidth=1.5)
for j,boxes in enumerate(bp['boxes']):
    boxes.set(facecolor='C'+str(j))
    ax[1].text(j+1,median_data[j]+1, '%.1f' % median_data[j], horizontalalignment='center', fontsize=14)
    ax[1].text(j+1+0.35,median_data[j]+1, ('n=%.d' % int(n_data[j])), ha='center', va='center', fontsize=12, rotation='vertical')
ax[1].set(ylabel='error [m]', ylim=sett['lims']);
fig.savefig( os.path.join(current_dir, 'examples','comparison_all_transects.jpg'), dpi=150)
