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
import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

# region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
polygon = SDS_tools.coords_from_kml('NARRA_polygon.kml')
#polygon = [[[151.301454, -33.700754],
#            [151.311453, -33.702075],
#            [151.307237, -33.739761],
#            [151.294220, -33.736329],
#            [151.301454, -33.700754]]]
            
# date range
dates = ['2017-12-01', '2018-02-01']

# satellite missions
sat_list = ['L8','S2']

# name of the site
sitename = 'NARRA'

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename
        }

#%% 2. Retrieve images

# retrieve satellite images from GEE
#metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
filepath = os.path.join(os.getcwd(), 'data', sitename)
with open(os.path.join(filepath, sitename + '_metadata' + '.pkl'), 'rb') as f:
    metadata = pickle.load(f) 
    
#%% 3. Batch shoreline detection
    
# settings for the shoreline extraction
settings = { 
    # general parameters:
    'cloud_thresh': 0.2,        # threshold on maximum cloud cover
    'output_epsg': 28356,       # epsg code of spatial reference system desired for the output   
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection to the user for validation

    # add the inputs defined previously
    'inputs': inputs,
    
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
}

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_manual(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100        

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

# plot the mapped shorelines
fig = plt.figure()
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend()
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    
fig.set_size_inches([15.76,  8.52])

#%% 4. Shoreline analysis

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(os.getcwd(), 'data', sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f) 

# now we have to define cross-shore transects over which to quantify the shoreline changes
# each transect is defined by two points, its origin and a second point that defines its orientation
# the parameter transect length determines how far from the origin the transect will span
settings['transect_length'] = 500 

# there are 3 options to create the transects:
# - option 1: draw the shore-normal transects along the beach
# - option 2: load the transect coordinates from a .kml file
# - option 3: create the transects manually by providing the coordinates

# option 1: draw origin of transect first and then a second point to define the orientation
transects = SDS_transects.draw_transects(output, settings)
    
# option 2: load the transects from a KML file
kml_file = 'NARRA_transects.kml'
transects = SDS_transects.load_transects_from_kml(kml_file)

# option 3: create the transects by manually providing the coordinates of two points 
#transects = dict([])
#transects['Transect 1'] = np.array([[342836, 6269215], [343315, 6269071]])
#transects['Transect 2'] = np.array([[342482, 6268466], [342958, 6268310]])
#transects['Transect 3'] = np.array([[342185, 6267650], [342685, 6267641]])
   
#%% intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
settings['along_dist'] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings) 

# plot the time-series
from matplotlib import gridspec
fig = plt.figure()
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-50,50])
    if not i == len(cross_distance.keys()):
        ax.set_xticks = []
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-^', markersize=6)
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95,'Transect ' + key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    
fig.set_size_inches([15.76,  8.52])