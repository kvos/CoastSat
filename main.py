#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# load modules
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools

# region of interest (longitude, latitude), can also be loaded from a .kml polygon
polygon = SDS_tools.coords_from_kml('NARRA.kml')
#polygon = [[[151.301454, -33.700754],
#            [151.311453, -33.702075],
#            [151.307237, -33.739761],
#            [151.294220, -33.736329],
#            [151.301454, -33.700754]]]
            
# date range
dates = ['2017-12-01', '2018-01-01']

# satellite missions
sat_list = ['S2']

# name of the site
sitename = 'NARRA'

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename
        }

# retrieve satellite images from GEE
#metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
filepath = os.path.join(os.getcwd(), 'data', sitename)
with open(os.path.join(filepath, sitename + '_metadata' + '.pkl'), 'rb') as f:
    metadata = pickle.load(f)   
#%%
# settings for the shoreline extraction
settings = {
       
    # general parameters:
    'cloud_thresh': 0.2,        # threshold on maximum cloud cover
    'output_epsg': 28356,       # epsg code of spatial reference system desired for the output
       
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in pixels) of disk for buffer around sandy pixels
    'min_length_sl': 200,       # minimum length of shoreline perimeter to be kept 
    
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection and lets the user 
                                # decide which ones are correct and which ones are false due to
                                # the presence of clouds 
    # add the inputs 
    'inputs': inputs
}


# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
#SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_manual(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100        

# extract shorelines from all images (also saves output.pkl and output.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

#%%
# basic figure plotting the mapped shorelines
plt.figure()
plt.axis('equal')
plt.xlabel('Eastings [m]')
plt.ylabel('Northings [m]')
for satname in output.keys():
    if satname == 'meta':
        continue
    for i in range(len(output[satname]['shoreline'])):
        sl = output[satname]['shoreline'][i]
        date = output[satname]['timestamp'][i]
        plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend()
        
        