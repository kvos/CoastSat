#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# load modules
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import SDS_download, SDS_preprocess, SDS_shoreline


# define the area of interest (longitude, latitude)
polygon = [[[151.301454, -33.700754],
            [151.311453, -33.702075],
            [151.307237, -33.739761],
            [151.294220, -33.736329],
            [151.301454, -33.700754]]]
            
# define dates of interest
dates = ['2017-12-01', '2018-01-01']

# define satellite missions
sat_list = ['L5', 'L7', 'L8', 'S2']

# give a name to the site
sitename = 'NARRA'

# download satellite images (also saves metadata.pkl)
#SDS_download.get_images(sitename, polygon, dates, sat_list)

# load metadata structure (contains information on the downloaded satellite images and is created
# after all images have been successfully downloaded)
filepath = os.path.join(os.getcwd(), 'data', sitename)
with open(os.path.join(filepath, sitename + '_metadata' + '.pkl'), 'rb') as f:
    metadata = pickle.load(f)

# parameters and settings
settings = {
    'sitename': sitename,
       
    # general parameters:
    'cloud_thresh': 0.5,         # threshold on maximum cloud cover
    'output_epsg': 28356,        # epsg code of the desired output spatial reference system
       
    # shoreline detection parameters:
    'min_beach_size': 20,        # minimum number of connected pixels for a beach
    'buffer_size': 7,            # radius (in pixels) of disk for buffer around sandy pixels
    'min_length_sl': 200,       # minimum length of shoreline perimeter to be kept 
    'max_dist_ref': 100,        # max distance (in meters) allowed from a reference shoreline
    
    # quality control:
    'check_detection': True     # if True, shows each shoreline detection and lets the user 
                                # decide which ones are correct and which ones are false due to
                                # the presence of clouds
}

# preprocess images (cloud masking, pansharpening/down-sampling)
SDS_preprocess.preprocess_all_images(metadata, settings)

# create a reference shoreline (used to identify outliers and false detections)
settings['refsl'] = SDS_preprocess.get_reference_sl(metadata, settings)

# extract shorelines from all images (also saves output.pkl)
out = SDS_shoreline.extract_shorelines(metadata, settings)

# plot shorelines
plt.figure()
plt.axis('equal')
plt.xlabel('Eastings [m]')
plt.ylabel('Northings [m]')
for satname in out.keys():
    if satname == 'meta':
        continue
    for i in range(len(out[satname]['shoreline'])):
        sl = out[satname]['shoreline'][i]
        date = out[satname]['timestamp'][i]
        plt.plot(sl[:, 0], sl[:, 1], '-', label=date.strftime('%d-%m-%Y'))
plt.legend()
