#==========================================================#
# Create a classifier for satellite images
#==========================================================#

# load modules
import os
import pickle
import warnings
import numpy as np
import matplotlib.cm as cm
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pylab import ginput

import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_classification

filepath_sites = os.path.join(os.getcwd(), 'polygons')
sites = os.listdir(filepath_sites)

for site in sites:

    polygon = SDS_tools.coords_from_kml(os.path.join(filepath_sites,site))
    
    # load Sentinel-2 images
    inputs = {
        'polygon': polygon,
        'dates': ['2016-10-01', '2016-11-01'],
        'sat_list': ['S2'],
        'sitename': site[:site.find('.')]
            }
    
    satname = inputs['sat_list'][0]
    
    metadata = SDS_download.get_images(inputs)
    metadata = SDS_download.remove_cloudy_images(metadata,inputs,0.2)
    filepath = os.path.join(os.getcwd(), 'data', inputs['sitename'])
    with open(os.path.join(filepath, inputs['sitename'] + '_metadata_' + satname + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    #with open(os.path.join(filepath, inputs['sitename'] + '_metadata_' + satname + '.pkl'), 'rb') as f:
    #    metadata = pickle.load(f)
        
    # settings needed to run the shoreline extraction
    settings = {
           
        # general parameters:
        'cloud_thresh': 0.1,         # threshold on maximum cloud cover
        'output_epsg': 28356,        # epsg code of spatial reference system desired for the output
           
        # shoreline detection parameters:
        'min_beach_size': 20,        # minimum number of connected pixels for a beach
        'buffer_size': 7,            # radius (in pixels) of disk for buffer around sandy pixels
        'min_length_sl': 200,       # minimum length of shoreline perimeter to be kept 
        'max_dist_ref': 100,        # max distance (in meters) allowed from a reference shoreline
        
        # quality control:
        'check_detection': True,    # if True, shows each shoreline detection and lets the user 
                                    # decide which ones are correct and which ones are false due to
                                    # the presence of clouds 
        # also add the inputs 
        'inputs': inputs
    }
    # preprocess images (cloud masking, pansharpening/down-sampling)
    SDS_preprocess.preprocess_all_images(metadata, settings)
        
    training_data = dict([])
    training_data['sand'] = dict([])
    training_data['swash'] = dict([])
    training_data['water'] = dict([])
    training_data['land'] = dict([])
    
    # read images
    filepath = SDS_tools.get_filepath(inputs,satname)
    filenames = metadata[satname]['filenames']
    
    for i in range(len(filenames)):
    
        fn = SDS_tools.get_filenames(filenames[i],filepath,satname)
        im_ms, georef, cloud_mask, im20, imQA = SDS_preprocess.preprocess_single(fn,satname)
        
        nrow = im_ms.shape[0]
        ncol = im_ms.shape[1]
        
        im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
        plt.figure()
        mng = plt.get_current_fig_manager()                                         
        mng.window.showMaximized()
        plt.imshow(im_RGB)
        plt.axis('off')
        
        # Digitize sandy pixels
        plt.title('Digitize SAND pixels', fontweight='bold', fontsize=15)
        pt = ginput(n=1000, timeout=100000, show_clicks=True)
        
        if len(pt) > 0:
            pt = np.round(pt).astype(int)    
            im_sand = np.zeros((nrow,ncol))
            for k in range(len(pt)):
                im_sand[pt[k,1],pt[k,0]] = 1
                im_RGB[pt[k,1],pt[k,0],0] = 1
                im_RGB[pt[k,1],pt[k,0],1] = 0
                im_RGB[pt[k,1],pt[k,0],2] = 0
            im_sand = im_sand.astype(bool)          
            features = SDS_classification.calculate_features(im_ms, cloud_mask, im_sand)
        else:
            im_sand = np.zeros((nrow,ncol)).astype(bool)
            features = []     
        training_data['sand'][filenames[i]] = {'pixels':im_sand,'features':features}
        
        # Digitize swash pixels
        plt.title('Digitize SWASH pixels', fontweight='bold', fontsize=15)
        plt.draw()
        pt = ginput(n=1000, timeout=100000, show_clicks=True)
    
        if len(pt) > 0:
            pt = np.round(pt).astype(int)
            im_swash = np.zeros((nrow,ncol))
            for k in range(len(pt)):
                im_swash[pt[k,1],pt[k,0]] = 1
                im_RGB[pt[k,1],pt[k,0],0] = 0
                im_RGB[pt[k,1],pt[k,0],1] = 1
                im_RGB[pt[k,1],pt[k,0],2] = 0
            im_swash = im_swash.astype(bool)  
            features = SDS_classification.calculate_features(im_ms, cloud_mask, im_swash)
        else:
            im_swash = np.zeros((nrow,ncol)).astype(bool)
            features = []
        training_data['swash'][filenames[i]] = {'pixels':im_swash,'features':features}
    
        # Digitize rectangle containig water pixels
        plt.title('Click 2 points to draw a rectange in the WATER', fontweight='bold', fontsize=15)
        plt.draw()
        pt = ginput(n=2, timeout=100000, show_clicks=True)
        if len(pt) > 0:
            pt = np.round(pt).astype(int) 
            idx_row = np.arange(np.min(pt[:,1]),np.max(pt[:,1])+1,1) 
            idx_col = np.arange(np.min(pt[:,0]),np.max(pt[:,0])+1,1) 
            xx, yy = np.meshgrid(idx_row,idx_col, indexing='ij')
            rows = xx.reshape(xx.shape[0]*xx.shape[1])
            cols = yy.reshape(yy.shape[0]*yy.shape[1])
            im_water = np.zeros((nrow,ncol)).astype(bool)
            for k in range(len(rows)):
                im_water[rows[k],cols[k]] = 1
                im_RGB[rows[k],cols[k],0] = 0
                im_RGB[rows[k],cols[k],1] = 0
                im_RGB[rows[k],cols[k],2] = 1
            im_water = im_water.astype(bool)        
            features = SDS_classification.calculate_features(im_ms, cloud_mask, im_water)
        else:
            im_water = np.zeros((nrow,ncol)).astype(bool)
            features = []     
        training_data['water'][filenames[i]] = {'pixels':im_water,'features':features}
        
        # Digitize rectangle containig land pixels
        plt.title('Click 2 points to draw a rectange in the LAND', fontweight='bold', fontsize=15)
        plt.draw()
        pt = ginput(n=2, timeout=100000, show_clicks=True)
        plt.close()
        if len(pt) > 0:
            pt = np.round(pt).astype(int)  
            idx_row = np.arange(np.min(pt[:,1]),np.max(pt[:,1])+1,1) 
            idx_col = np.arange(np.min(pt[:,0]),np.max(pt[:,0])+1,1) 
            xx, yy = np.meshgrid(idx_row,idx_col, indexing='ij')
            rows = xx.reshape(xx.shape[0]*xx.shape[1])
            cols = yy.reshape(yy.shape[0]*yy.shape[1]) 
            im_land = np.zeros((nrow,ncol)).astype(bool)
            for k in range(len(rows)):
                im_land[rows[k],cols[k]] = 1
                im_RGB[rows[k],cols[k],0] = 1
                im_RGB[rows[k],cols[k],1] = 1
                im_RGB[rows[k],cols[k],2] = 0
            im_land = im_land.astype(bool) 
            features = SDS_classification.calculate_features(im_ms, cloud_mask, im_land)
        else:
            im_land = np.zeros((nrow,ncol)).astype(bool)
            features = []  
        training_data['land'][filenames[i]] = {'pixels':im_land,'features':features}
         
        plt.figure()
        plt.title('Classified image')
        plt.imshow(im_RGB)
        
    # save training data for each site
    filepath = os.path.join(os.getcwd(), 'data', inputs['sitename'])
    with open(os.path.join(filepath, inputs['sitename'] + '_training_' + satname + '.pkl'), 'wb') as f:
        pickle.dump(training_data, f)
#%%

## load Landsat 5 images
#inputs = {
#    'polygon': polygon,
#    'dates': ['1987-01-01', '1988-01-01'],
#    'sat_list': ['L5'],
#    'sitename': site[:site.find('.')]
#        }
#metadata = SDS_download.get_images(inputs)
#
## load Landsat 7 images
#inputs = {
#    'polygon': polygon,
#    'dates': ['2001-01-01', '2002-01-01'],
#    'sat_list': ['L7'],
#    'sitename': site[:site.find('.')]
#        }
#metadata = SDS_download.get_images(inputs)
#
## load Landsat 8 images
#inputs = {
#    'polygon': polygon,
#    'dates': ['2014-01-01', '2015-01-01'],
#    'sat_list': ['L8'],
#    'sitename': site[:site.find('.')]
#        }
#metadata = SDS_download.get_images(inputs)


#%% clean the Landsat collections

#import ee
#from datetime import datetime, timedelta
#import pytz
#import copy
#ee.Initialize()
#site = sites[0]
#dates = ['2017-12-01', '2017-12-25']
#polygon = SDS_tools.coords_from_kml(os.path.join(filepath_sites,site))
## Landsat collection
#input_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')
## filter by location and dates
#flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(inputs['dates'][0],inputs['dates'][1])
## get all images in the filtered collection
#im_all = flt_col.getInfo().get('features')
#cloud_cover = [_['properties']['CLOUD_COVER'] for _ in im_all]
#if np.any([_ > 90 for _ in cloud_cover]):
#    idx_delete = np.where([_ > 90 for _ in cloud_cover])[0]
#    im_all_cloud = [x for k,x in enumerate(im_all) if k not in idx_delete]


#%% clean the S2 collection

#import ee
#from datetime import datetime, timedelta
#import pytz
#import copy
#ee.Initialize()
## Sentinel2 collection
#input_col = ee.ImageCollection('COPERNICUS/S2')
## filter by location and dates
#flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(inputs['dates'][0],inputs['dates'][1])
## get all images in the filtered collection
#im_all = flt_col.getInfo().get('features')
#
## remove duplicates (there are many in S2 collection)
## timestamps
#timestamps = [datetime.fromtimestamp(_['properties']['system:time_start']/1000, tz=pytz.utc) for _ in im_all]
## utm zones
#utm_zones = np.array([int(_['bands'][0]['crs'][5:]) for _ in im_all])
#utm_zone_selected =  np.max(np.unique(utm_zones))
#idx_all = np.arange(0,len(im_all),1)
#idx_covered = np.ones(len(im_all)).astype(bool)
#idx_delete = []
#i = 0
#while 1:
#    same_time = np.abs([(timestamps[i]-_).total_seconds() for _ in timestamps]) < 60*60*24
#    idx_same_time = np.where(same_time)[0]
#    same_utm = utm_zones == utm_zone_selected
#    idx_temp = np.where([same_time[j] == True and same_utm[j] == False for j in idx_all])[0]
#    idx_keep = idx_same_time[[_ not in idx_temp for _ in idx_same_time ]]
#    if len(idx_keep) > 2: # if more than 2 images with same date and same utm, drop the last one
#       idx_temp = np.append(idx_temp,idx_keep[-1])
#    for j in idx_temp:
#        idx_delete.append(j)
#    idx_covered[idx_same_time] = False
#    if np.any(idx_covered):
#        i = np.where(idx_covered)[0][0]
#    else:
#        break
#im_all_updated = [x for k,x in enumerate(im_all) if k not in idx_delete]
#
## remove very cloudy images (>90% cloud)
#cloud_cover = [_['properties']['CLOUDY_PIXEL_PERCENTAGE'] for _ in im_all_updated]
#if np.any([_ > 90 for _ in cloud_cover]):
#    idx_delete = np.where([_ > 90 for _ in cloud_cover])[0]
#    im_all_cloud = [x for k,x in enumerate(im_all_updated) if k not in idx_delete]


