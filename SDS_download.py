"""This module contains all the functions needed to download the satellite images from GEE
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# Initial settings
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import ee
from urllib.request import urlretrieve
from datetime import datetime
import pytz
import pickle
import zipfile

# initialise connection with GEE server
ee.Initialize()

# Functions

def download_tif(image, polygon, bandsId, filepath):
    """
    Downloads a .TIF image from the ee server and stores it in a temp file
        
    Arguments:
    -----------
        image: ee.Image
            Image object to be downloaded
        polygon: list
            polygon containing the lon/lat coordinates to be extracted
            longitudes in the first column and latitudes in the second column
        bandsId: list of dict
            list of bands to be downloaded
        filepath: location where the temporary file should be saved
            
    """
    
    url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
        'image': image.serialize(),
        'region': polygon,
        'bands': bandsId,
        'filePerBand': 'false',
        'name': 'data',
        }))
    local_zip, headers = urlretrieve(url)
    with zipfile.ZipFile(local_zip) as local_zipfile:
        return local_zipfile.extract('data.tif', filepath)


def get_images(sitename,polygon,dates,sat):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 covering the given 
    polygon and acquired during the given dates. The images are organised in subfolders and divided
    by satellite mission and pixel resolution.
    
    KV WRL 2018
        
    Arguments:
    -----------
        sitename: str
            String containig the name of the site
        polygon: list
            polygon containing the lon/lat coordinates to be extracted
            longitudes in the first column and latitudes in the second column
        dates: list of str
            list that contains 2 strings with the initial and final dates in format 'yyyy-mm-dd'
            e.g. ['1987-01-01', '2018-01-01']
        sat: list of str
            list that contains the names of the satellite missions to include 
            e.g. ['L5', 'L7', 'L8', 'S2']
            
    """
    
    # format in which the images are downloaded
    suffix = '.tif'
 
    # initialise metadata dictionnary (stores timestamps and georefencing accuracy of each image)       
    metadata = dict([])
    
    # create directories
    try:
        os.makedirs(os.path.join(os.getcwd(), 'data',sitename))
    except:
        print('')
        
    #=============================================================================================#
    # download L5 images
    #=============================================================================================#
    
    if 'L5' in sat or 'Landsat5' in sat:
        
        satname = 'L5'
        # create a subfolder to store L5 images
        filepath = os.path.join(os.getcwd(), 'data', sitename, satname, '30m')
        try:
            os.makedirs(filepath)
        except:
            print('')
        
        # Landsat 5 collection
        input_col = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA')
        # filter by location and dates
        flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
        # get all images in the filtered collection
        im_all = flt_col.getInfo().get('features')
        # print how many images there are for the user
        n_img = flt_col.size().getInfo()
        print('Number of ' + satname + ' images covering ' + sitename + ':', n_img)
       
       # loop trough images
        timestamps = []
        acc_georef = []
        all_names = []
        im_epsg = []
        for i in range(n_img):
            
            # find each image in ee database
            im = ee.Image(im_all[i].get('id'))
            # read metadata
            im_dic = im.getInfo()
            # get bands
            im_bands = im_dic.get('bands')
            # get time of acquisition (UNIX time)
            t = im_dic['properties']['system:time_start']
            # convert to datetime
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            timestamps.append(im_timestamp)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            # get EPSG code of reference system
            im_epsg.append(int(im_dic['bands'][0]['crs'][5:]))
            # get geometric accuracy
            try:
                acc_georef.append(im_dic['properties']['GEOMETRIC_RMSE_MODEL'])
            except:
                # default value of accuracy (RMSE = 12m)
                acc_georef.append(12)   
                print('No geometric rmse model property')
            # delete dimensions key from dictionnary, otherwise the entire image is extracted
            for j in range(len(im_bands)): del im_bands[j]['dimensions']
            # bands for L5
            ms_bands = [im_bands[0], im_bands[1], im_bands[2], im_bands[3], im_bands[4], im_bands[7]]
            # filenames for the images
            filename = im_date + '_' + satname + '_' + sitename + suffix
            # if two images taken at the same date add 'dup' in the name
            if any(filename in _ for _ in all_names):
                filename = im_date + '_' + satname + '_' + sitename + '_dup' + suffix 
            all_names.append(filename)
            # download .TIF image
            local_data = download_tif(im, polygon, ms_bands, filepath)
            # update filename
            os.rename(local_data, os.path.join(filepath, filename))
            print(i, end='..')
        
        # sort timestamps and georef accuracy (dowloaded images are sorted by date in directory)
        timestamps_sorted = sorted(timestamps)
        idx_sorted = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        acc_georef_sorted = [acc_georef[j] for j in idx_sorted]
        im_epsg_sorted = [im_epsg[j] for j in idx_sorted]
        # save into dict
        metadata[satname] = {'dates':timestamps_sorted, 'acc_georef':acc_georef_sorted,
                'epsg':im_epsg_sorted}   
        print('Finished with ' + satname)
    
    
    
    #=============================================================================================#
    # download L7 images
    #=============================================================================================#
    
    if 'L7' in sat or 'Landsat7' in sat:
        
        satname = 'L7'
        # create subfolders (one for 30m multispectral bands and one for 15m pan bands)
        filepath = os.path.join(os.getcwd(), 'data', sitename, 'L7')
        filepath_pan = os.path.join(filepath, 'pan')
        filepath_ms = os.path.join(filepath, 'ms')
        try:
            os.makedirs(filepath_pan)
            os.makedirs(filepath_ms)
        except:
            print('')
         
        # landsat 7 collection
        input_col = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT_TOA')
        # filter by location and dates
        flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
        # get all images in the filtered collection
        im_all = flt_col.getInfo().get('features')
        # print how many images there are for the user
        n_img = flt_col.size().getInfo()
        print('Number of ' + satname + ' images covering ' + sitename + ':', n_img)
        
        # loop trough images
        timestamps = []
        acc_georef = []
        all_names = []
        im_epsg = []
        for i in range(n_img):
            
            # find each image in ee database
            im = ee.Image(im_all[i].get('id'))
            # read metadata
            im_dic = im.getInfo()
            # get bands
            im_bands = im_dic.get('bands')
            # get time of acquisition (UNIX time)
            t = im_dic['properties']['system:time_start']
            # convert to datetime
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            timestamps.append(im_timestamp)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            # get EPSG code of reference system
            im_epsg.append(int(im_dic['bands'][0]['crs'][5:]))
            # get geometric accuracy
            try:
                acc_georef.append(im_dic['properties']['GEOMETRIC_RMSE_MODEL'])
            except:
                # default value of accuracy (RMSE = 12m)
                acc_georef.append(12)   
                print('No geometric rmse model property')
            # delete dimensions key from dictionnary, otherwise the entire image is extracted
            for j in range(len(im_bands)): del im_bands[j]['dimensions']   
            # bands for L7
            pan_band = [im_bands[8]]
            ms_bands = [im_bands[0], im_bands[1], im_bands[2], im_bands[3], im_bands[4], im_bands[9]] 
            # filenames for the images
            filename_pan = im_date + '_' + satname + '_' + sitename + '_pan' + suffix
            filename_ms = im_date + '_' + satname + '_' + sitename + '_ms' + suffix  
            # if two images taken at the same date add 'dup' in the name
            if any(filename_pan in _ for _ in all_names):
                filename_pan = im_date + '_' + satname + '_' + sitename + '_pan' + '_dup' + suffix
                filename_ms = im_date + '_' + satname + '_' + sitename + '_ms' + '_dup' + suffix 
            all_names.append(filename_pan)        
            # download .TIF image
            local_data_pan = download_tif(im, polygon, pan_band, filepath_pan)
            local_data_ms = download_tif(im, polygon, ms_bands, filepath_ms)
            # update filename
            os.rename(local_data_pan, os.path.join(filepath_pan, filename_pan))
            os.rename(local_data_ms, os.path.join(filepath_ms, filename_ms))
            print(i, end='..')  
            
        # sort timestamps and georef accuracy (dowloaded images are sorted by date in directory)
        timestamps_sorted = sorted(timestamps)
        idx_sorted = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        acc_georef_sorted = [acc_georef[j] for j in idx_sorted]
        im_epsg_sorted = [im_epsg[j] for j in idx_sorted]
        # save into dict
        metadata[satname] = {'dates':timestamps_sorted, 'acc_georef':acc_georef_sorted,
                'epsg':im_epsg_sorted}
        print('Finished with ' + satname)
        
        
    #=============================================================================================#
    # download L8 images
    #=============================================================================================#
    
    if 'L8' in sat or 'Landsat8' in sat:

        satname = 'L8'  
        # create subfolders (one for 30m multispectral bands and one for 15m pan bands)
        filepath = os.path.join(os.getcwd(), 'data', sitename, 'L8')
        filepath_pan = os.path.join(filepath, 'pan')
        filepath_ms = os.path.join(filepath, 'ms')
        try:
            os.makedirs(filepath_pan)
            os.makedirs(filepath_ms)
        except:
            print('')
            
        # landsat 8 collection
        input_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')
        # filter by location and dates
        flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
        # get all images in the filtered collection
        im_all = flt_col.getInfo().get('features')
        # print how many images there are for the user
        n_img = flt_col.size().getInfo()
        print('Number of ' + satname + ' images covering ' + sitename + ':', n_img)
        
       # loop trough images
        timestamps = []
        acc_georef = []
        all_names = []
        im_epsg = []
        for i in range(n_img):
            
            # find each image in ee database
            im = ee.Image(im_all[i].get('id'))
            # read metadata
            im_dic = im.getInfo()
            # get bands
            im_bands = im_dic.get('bands')
            # get time of acquisition (UNIX time)
            t = im_dic['properties']['system:time_start']
            # convert to datetime
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            timestamps.append(im_timestamp)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            # get EPSG code of reference system
            im_epsg.append(int(im_dic['bands'][0]['crs'][5:]))
            # get geometric accuracy
            try:
                acc_georef.append(im_dic['properties']['GEOMETRIC_RMSE_MODEL'])
            except:
                # default value of accuracy (RMSE = 12m)
                acc_georef.append(12)   
                print('No geometric rmse model property')
            # delete dimensions key from dictionnary, otherwise the entire image is extracted
            for j in range(len(im_bands)): del im_bands[j]['dimensions']   
            # bands for L8    
            pan_band = [im_bands[7]]
            ms_bands = [im_bands[1], im_bands[2], im_bands[3], im_bands[4], im_bands[5], im_bands[11]]
            # filenames for the images
            filename_pan = im_date + '_' + satname + '_' + sitename + '_pan' + suffix
            filename_ms = im_date + '_' + satname + '_' + sitename + '_ms' + suffix  
            # if two images taken at the same date add 'dup' in the name
            if any(filename_pan in _ for _ in all_names):
                filename_pan = im_date + '_' + satname + '_' + sitename + '_pan' + '_dup' + suffix
                filename_ms = im_date + '_' + satname + '_' + sitename + '_ms' + '_dup' + suffix 
            all_names.append(filename_pan)        
            # download .TIF image
            local_data_pan = download_tif(im, polygon, pan_band, filepath_pan)
            local_data_ms = download_tif(im, polygon, ms_bands, filepath_ms)
            # update filename
            os.rename(local_data_pan, os.path.join(filepath_pan, filename_pan))
            os.rename(local_data_ms, os.path.join(filepath_ms, filename_ms))
            print(i, end='..')
    
        # sort timestamps and georef accuracy (dowloaded images are sorted by date in directory)
        timestamps_sorted = sorted(timestamps)
        idx_sorted = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        acc_georef_sorted = [acc_georef[j] for j in idx_sorted]
        im_epsg_sorted = [im_epsg[j] for j in idx_sorted]
        
        metadata[satname] = {'dates':timestamps_sorted, 'acc_georef':acc_georef_sorted,
                'epsg':im_epsg_sorted}
        print('Finished with ' + satname)

    #=============================================================================================#
    # download S2 images
    #=============================================================================================#
    
    if 'S2' in sat or 'Sentinel2' in sat:

        satname = 'S2' 
        # create subfolders for the 10m, 20m and 60m multipectral bands
        filepath = os.path.join(os.getcwd(), 'data', sitename, 'S2')
        try:
            os.makedirs(os.path.join(filepath, '10m'))
            os.makedirs(os.path.join(filepath, '20m'))
            os.makedirs(os.path.join(filepath, '60m'))
        except:
            print('')
    
        # Sentinel2 collection
        input_col = ee.ImageCollection('COPERNICUS/S2')
        # filter by location and dates
        flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
        # get all images in the filtered collection
        im_all = flt_col.getInfo().get('features')
        # print how many images there are
        n_img = flt_col.size().getInfo()
        print('Number of ' + satname + ' images covering ' + sitename + ':', n_img)    
    
       # loop trough images
        timestamps = []
        acc_georef = []
        all_names = []
        im_epsg = []
        for i in range(n_img):
            
            # find each image in ee database
            im = ee.Image(im_all[i].get('id'))
            # read metadata
            im_dic = im.getInfo()
            # get bands
            im_bands = im_dic.get('bands')
            # get time of acquisition (UNIX time)
            t = im_dic['properties']['system:time_start']
            # convert to datetime
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')   
            # delete dimensions key from dictionnary, otherwise the entire image is extracted
            for j in range(len(im_bands)): del im_bands[j]['dimensions'] 
            # bands for S2
            bands10 = [im_bands[1], im_bands[2], im_bands[3], im_bands[7]]
            bands20 = [im_bands[11]]
            bands60 = [im_bands[15]]    
            # filenames for images
            filename10 = im_date + '_' + satname + '_' + sitename + '_' + '10m' + suffix
            filename20 = im_date + '_' + satname + '_' + sitename + '_' + '20m' + suffix
            filename60 = im_date + '_' + satname + '_' + sitename + '_' + '60m' + suffix
            # if two images taken at the same date skip the second image (they are the same)
            if any(filename10 in _ for _ in all_names):
                continue
            all_names.append(filename10)  
            # download .TIF image and update filename
            local_data = download_tif(im, polygon, bands10, os.path.join(filepath, '10m'))
            os.rename(local_data, os.path.join(filepath, '10m', filename10))
            local_data = download_tif(im, polygon, bands20, os.path.join(filepath, '20m'))
            os.rename(local_data, os.path.join(filepath, '20m', filename20))
            local_data = download_tif(im, polygon, bands60, os.path.join(filepath, '60m'))
            os.rename(local_data, os.path.join(filepath, '60m', filename60))
    
            # save timestamp, epsg code and georeferencing accuracy (1 if passed 0 if not passed)
            timestamps.append(im_timestamp)
            im_epsg.append(int(im_dic['bands'][0]['crs'][5:]))
            try:
                if im_dic['properties']['GEOMETRIC_QUALITY_FLAG'] == 'PASSED':
                    acc_georef.append(1)
                else:
                    acc_georef.append(0)
            except:
                acc_georef.append(0)
            print(i, end='..')
    
        # sort timestamps and georef accuracy (dowloaded images are sorted by date in directory)
        timestamps_sorted = sorted(timestamps)
        idx_sorted = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        acc_georef_sorted = [acc_georef[j] for j in idx_sorted]
        im_epsg_sorted = [im_epsg[j] for j in idx_sorted]
        
        metadata[satname] = {'dates':timestamps_sorted, 'acc_georef':acc_georef_sorted,
                'epsg':im_epsg_sorted} 
        print('Finished with ' + satname)

    # save metadata dict
    filepath = os.path.join(os.getcwd(), 'data', sitename)
    with open(os.path.join(filepath, sitename + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f) 