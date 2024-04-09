"""
This module contains utilities to work with satellite images
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pdb

# other modules
from osgeo import gdal, osr
import geopandas as gpd
from shapely import geometry
import skimage.transform as transform
from astropy.convolution import convolve
import pytz
from datetime import datetime, timedelta
from scipy import stats, interpolate
import pyproj
import pandas as pd
import imageio

###################################################################################################
# COORDINATES CONVERSION FUNCTIONS
###################################################################################################

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted

def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define transformer
    proj = pyproj.Transformer.from_crs(epsg_in, epsg_out, always_xy=True)
    
    # transform points
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            x,y = proj.transform(arr[:,0], arr[:,1])
            arr_converted = np.transpose(np.array([x,y]))
            points_converted.append(arr_converted)
    elif type(points) is np.ndarray:
        x,y = proj.transform(points[:,0], points[:,1])
        points_converted = np.transpose(np.array([x,y]))
    else:
        raise Exception('invalid input type')

    return points_converted

###################################################################################################
# IMAGE ANALYSIS FUNCTIONS
###################################################################################################
    
def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
        
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd
    
def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of 
    specified radius. Uses astropy's convolution library'
    
    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation. 
        For example, radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std with uniform filters
    win_mean = convolve(image_padded, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_sqr_mean = convolve(image_padded**2, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std

def mask_raster(fn, mask):
    """
    Masks a .tif raster using GDAL.
    
    Arguments:
    -----------
    fn: str
        filepath + filename of the .tif raster
    mask: np.array
        array of boolean where True indicates the pixels that are to be masked
        
    Returns:    
    -----------
    Overwrites the .tif file directly
        
    """ 
    
    # open raster
    raster = gdal.Open(fn, gdal.GA_Update)
    # mask raster
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i+1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)
    # close dataset and flush cache
    raster = None

def get_image_bounds(fn):
    """
    Returns a polygon with the bounds of the image in the .tif file
     
    KV WRL 2020

    Arguments:
    -----------
    fn: str
        path to the image (.tif file)         
                
    Returns:    
    -----------
    bounds_polygon: shapely.geometry.Polygon
        polygon with the image bounds
        
    """
    
    # nested functions to get the extent 
    # copied from https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
    def GetExtent(gt,cols,rows):
        'Return list of corner coordinates from a geotransform'
        ext=[]
        xarr=[0,cols]
        yarr=[0,rows]
        for px in xarr:
            for py in yarr:
                x=gt[0]+(px*gt[1])+(py*gt[2])
                y=gt[3]+(px*gt[4])+(py*gt[5])
                ext.append([x,y])
            yarr.reverse()
        return ext
    
    # load .tif file and get bounds
    if not os.path.exists(fn):
        raise FileNotFoundError(f"{fn}")
    data = gdal.Open(fn, gdal.GA_ReadOnly)
    # Check if data is null meaning the open failed
    if data is None:
        print("TIF file: ",fn, "cannot be opened" )
        os.remove(fn)
        raise AttributeError
    else:
        gt = data.GetGeoTransform()
        cols = data.RasterXSize
        rows = data.RasterYSize
        ext = GetExtent(gt,cols,rows)
    
    return geometry.Polygon(ext)

def get_image_dimensions(image_path):
    "function to get image dimensions with GDAL"
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception("Failed to open the image file %s"%image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    dataset = None

    return width, height

###################################################################################################
# UTILITIES
###################################################################################################

def create_folder_structure(im_folder, satname):
    """
    Create the structure of subfolders for each satellite mission

    KV WRL 2018

    Arguments:
    -----------
    im_folder: str
        folder where the images are to be downloaded
    satname:
        name of the satellite mission

    Returns:
    -----------
    filepaths: list of str
        filepaths of the folders that were created
    """

    # one folder for the metadata (common to all satellites)
    filepaths = [os.path.join(im_folder, satname, 'meta')]
    # subfolders depending on satellite mission
    if satname == 'L5':
        filepaths.append(os.path.join(im_folder, satname, 'ms'))
        filepaths.append(os.path.join(im_folder, satname, 'mask'))
    elif satname in ['L7','L8','L9']:
        filepaths.append(os.path.join(im_folder, satname, 'ms'))
        filepaths.append(os.path.join(im_folder, satname, 'pan'))
        filepaths.append(os.path.join(im_folder, satname, 'mask'))
    elif satname in ['S2']:
        filepaths.append(os.path.join(im_folder, satname, 'ms'))
        filepaths.append(os.path.join(im_folder, satname, 'swir'))
        filepaths.append(os.path.join(im_folder, satname, 'mask'))
    # create the subfolders if they don't exist already
    for fp in filepaths:
        if not os.path.exists(fp): os.makedirs(fp)

    return filepaths

def get_filepath(inputs,satname):
    """
    Create filepath to the different folders containing the satellite images.
    
    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in 
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include: 
            ```
            sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']
            ```
        'filepath': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')
                
    Returns:    
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    
    """     
    
    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        fp_ms = os.path.join(filepath_data, sitename, satname, 'ms')
        fp_mask = os.path.join(filepath_data, sitename, satname, 'mask')
        filepath = [fp_ms, fp_mask]
    elif satname in ['L7','L8','L9']:
        # access downloaded Landsat 7 images
        fp_ms = os.path.join(filepath_data, sitename, satname, 'ms')
        fp_pan = os.path.join(filepath_data, sitename, satname, 'pan')
        fp_mask = os.path.join(filepath_data, sitename, satname, 'mask')
        filepath = [fp_ms, fp_pan, fp_mask]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        fp_ms = os.path.join(filepath_data, sitename, satname, 'ms')
        fp_swir = os.path.join(filepath_data, sitename, satname, 'swir')
        fp_mask = os.path.join(filepath_data, sitename, satname, 'mask')
        filepath = [fp_ms, fp_swir, fp_mask]
            
    return filepath
    
def get_filenames(filename, filepath, satname):
    """
    Creates filepath + filename for all the bands belonging to the same image.
    
    KV WRL 2018

    Arguments:
    -----------
    filename: str
        name of the downloaded satellite image as found in the metadata
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    satname: str
        short name of the satellite mission       
        
    Returns:    
    -----------
    fn: str or list of str
        contains the filepath + filenames to access the satellite image
        
    """     
    
    if satname == 'L5':
        fn_mask = filename.replace('ms.tif','mask.tif')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], fn_mask)]
    if satname in ['L7','L8','L9']:
        fn_pan = filename.replace('ms.tif','pan.tif')
        fn_mask = filename.replace('ms.tif','mask.tif')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], fn_pan),
              os.path.join(filepath[2], fn_mask)]
    if satname == 'S2':
        fn_swir = filename.replace('_ms','_swir')
        fn_mask = filename.replace('_ms','_mask')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], fn_swir),
              os.path.join(filepath[2], fn_mask)]
        
    return fn

def merge_output(output):
    """
    Function to merge the output dictionnary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.
    
    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates, organised by 
        satellite mission
    
    Returns:    
    -----------
    output_all: dict
        contains the extracted shorelines in a single list sorted by date
    
    """     
    
    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all['satname'] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all['satname'] = output_all['satname'] + [_ for _ in np.tile(satname,
                  len(output[satname]['dates']))]
    # sort chronologically
    idx_sorted = sorted(range(len(output_all['dates'])), key=output_all['dates'].__getitem__)
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all

def remove_duplicates(output):
    """
    Function to remove from the output dictionnary entries containing shorelines for 
    the same date and satellite mission. This happens when there is an overlap 
    between adjacent satellite images.
    
    KV WRL 2020
    
    Arguments:
    -----------
        output: dict
            contains output dict with shoreline and metadata
        
    Returns:    
    -----------
        output_no_duplicates: dict
            contains the updated dict where duplicates have been removed
        
    """
    # remove duplicates
    dates = output['dates'].copy()
    # find the pairs of images that are within 5 minutes of each other
    time_delta = 5*60 # 5 minutes in seconds
    pairs = []
    for i,date in enumerate(dates):
        # dummy value so it does not match it again
        dates[i] = pytz.utc.localize(datetime(1,1,1) + timedelta(days=i+1))
        # calculate time difference
        time_diff = np.array([np.abs((date - _).total_seconds()) for _ in dates])
        # find the matching times and add to pairs list
        boolvec = time_diff <= time_delta
        if np.sum(boolvec) == 0:
            continue
        else:
            idx_dup = np.where(boolvec)[0][0]
            pairs.append([i,idx_dup])
            
    # if there are duplicates, only keep the longest shoreline
    if len(pairs) > 0:
        # initialise variables
        output_no_duplicates = dict([])
        idx_remove = []
        # for each pair
        for pair in pairs:
            # check if any of the shorelines are empty
            empty_bool = [(len(output['shorelines'][_]) < 2) for _ in pair]
            if np.all(empty_bool): # if both empty remove both
                idx_remove.append(pair[0])
                idx_remove.append(pair[1])
            elif np.any(empty_bool): # if one empty remove that one
                idx_remove.append(pair[np.where(empty_bool)[0][0]])
            else: # remove the shorter shoreline and keep the longer one
                satnames = [output['satname'][_] for _ in pair]
                # keep Landsat 9 if it duplicates Landsat 7
                if 'L9' in satnames and 'L7' in satnames: 
                    idx_remove.append(pair[np.where([_ == 'L7' for _ in satnames])[0][0]])
                else: # keep the longest shorelines
                    sl0 = geometry.LineString(output['shorelines'][pair[0]]) 
                    sl1 = geometry.LineString(output['shorelines'][pair[1]])
                    if sl0.length >= sl1.length: idx_remove.append(pair[1])
                    else: idx_remove.append(pair[0])
        # create a new output structure with all the duplicates removed
        idx_remove = sorted(idx_remove)
        idx_all = np.linspace(0, len(dates)-1, len(dates)).astype(int)
        idx_keep = list(np.where(~np.isin(idx_all,idx_remove))[0])        
        for key in output.keys():
            output_no_duplicates[key] = [output[key][i] for i in idx_keep]
        print('%d duplicates' % len(idx_remove))
        return output_no_duplicates 
    else: 
        print('0 duplicates')
        return output

def remove_inaccurate_georef(output, accuracy):
    """
    Function to remove from the output dictionnary entries containing shorelines 
    that were mapped on images with inaccurate georeferencing:
        - RMSE > accuracy for Landsat images
        - failed geometric test for Sentinel images (flagged with -1)

    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding metadata
        accuracy: int
            minimum horizontal georeferencing accuracy (metres) for a shoreline to be accepted

    Returns:
    -----------
        output_filtered: dict
            contains the updated dictionnary

    """

    # find indices of shorelines to be removed
    idx = []
    for i in range(len(output['geoaccuracy'])):
        geoacc = output['geoaccuracy'][i]
        if geoacc in ['PASSED','FAILED']:
            if geoacc == 'PASSED':
                idx.append(i)
        else:
            if geoacc <= accuracy:
                idx.append(i)
    # idx = np.where(~(np.array(output['geoaccuracy']) >= accuracy))[0]
    output_filtered = dict([])
    for key in output.keys():
        output_filtered[key] = [output[key][i] for i in idx]
    print('%d bad georef' % (len(output['geoaccuracy']) - len(idx)))
    return output_filtered

def get_closest_datapoint(dates, dates_ts, values_ts):
    """
    Extremely efficient script to get closest data point to a set of dates from a very
    long time-series (e.g., 15-minutes tide data, or hourly wave data)
    
    Make sure that dates and dates_ts are in the same timezone (also aware or naive)
    
    KV WRL 2020

    Arguments:
    -----------
    dates: list of datetimes
        dates at which the closest point from the time-series should be extracted
    dates_ts: list of datetimes
        dates of the long time-series
    values_ts: np.array
        array with the values of the long time-series (tides, waves, etc...)
        
    Returns:    
    -----------
    values: np.array
        values corresponding to the input dates
        
    """
    
    # check if the time-series cover the dates
    if dates[0] < dates_ts[0] or dates[-1] > dates_ts[-1]: 
        raise Exception('Time-series do not cover the range of your input dates')
    
    # get closest point to each date (no interpolation)
    temp = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    for i,date in enumerate(dates):
        print('\rExtracting closest points: %d%%' % int((i+1)*100/len(dates)), end='')
        temp.append(values_ts[find(min(item for item in dates_ts if item > date), dates_ts)])
    values = np.array(temp)
    
    return values

###################################################################################################
# GEODATAFRAMES AND READ/WRITE GEOJSON
###################################################################################################
    
def polygon_from_geojson(fn):
    """
    Extracts coordinates from a .kml file.
    
    KV WRL 2023

    Arguments:
    -----------
    fn: str
        filepath + filename of the geojson file to be read          
                
    Returns:    
    -----------
    polygon: list
        coordinates extracted from the .geojson file
        
    """    
    
    # read .geojson file
    gdf = gpd.read_file(fn,driver='GeoJSON')
    coords = np.array(gdf.iloc[0]['geometry'].exterior.coords)
    polygon = [[[_[0], _[1]] for _ in coords]]
    return polygon

def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.
    
    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read          
                
    Returns:    
    -----------
    polygon: list
        coordinates extracted from the .kml file
        
    """    
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    # read coordinates
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    """  
    
    gdf = gpd.read_file(filename,driver='GeoJSON')
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
    print('%d transects have been loaded'%len(transects.keys()), end=' ')
    print('coordinates are in epsg:%d'%gdf.crs.to_epsg())

    return transects

def output_to_gdf(output, geomtype):
    """
    Saves the mapped shorelines as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes
    geomtype: str
        'lines' for LineString and 'points' for Multipoint geometry      
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes
  
    """    
     
    # loop through the mapped shorelines
    counter = 0
    gdf_all = None
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty 
        if len(output['shorelines'][i]) < 2:
            continue
        else:
            # save the geometry depending on the linestyle
            if geomtype == 'lines':
                geom = geometry.LineString(output['shorelines'][i])
            elif geomtype == 'points':
                coords = output['shorelines'][i]
                geom = geometry.MultiPoint([(coords[_,0], coords[_,1]) for _ in range(coords.shape[0])])
            else:
                raise Exception('geomtype %s is not an option, choose between lines or points'%geomtype)
            # save into geodataframe with attributes
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i,'date'] = output['dates'][i].strftime('%Y-%m-%d %H:%M:%S')
            gdf.loc[i,'satname'] = output['satname'][i]
            gdf.loc[i,'geoaccuracy'] = output['geoaccuracy'][i]
            gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = pd.concat([gdf_all, gdf])
            counter = counter + 1
            
    return gdf_all

def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame

        
    """  
       
    # loop through the mapped shorelines
    for i,key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i,'name'] = key
        # store into geodataframe
        if i == 0:
            gdf_all = gdf
        else:
            gdf_all = pd.concat([gdf_all, gdf])
            
    return gdf_all

def smallest_rectangle(polygon):
    """
    Converts a polygon to the smallest rectangle polygon with sides parallel
    to coordinate axes.
     
    KV WRL 2020

    Arguments:
    -----------
    polygon: list of coordinates 
        pair of coordinates for 5 vertices, in clockwise order,
        first and last points must match     
                
    Returns:    
    -----------
    polygon: list of coordinates
        smallest rectangle polygon
        
    """
    
    multipoints = geometry.Polygon(polygon[0])
    polygon_geom = multipoints.envelope
    coords_polygon = np.array(polygon_geom.exterior.coords)
    polygon_rect = [[[_[0], _[1]] for _ in coords_polygon]]
    return polygon_rect

###################################################################################################
# MAKE ANIMATIONS
###################################################################################################

def make_animation_mp4(filepath_images, fps, fn_out):
    "function to create an animation with the saved figures"
    with imageio.get_writer(fn_out, mode='I', fps=fps) as writer:
        filenames = os.listdir(filepath_images)
        # order chronologically
        filenames = np.sort(filenames)
        for i in range(len(filenames)):
            image = imageio.imread(os.path.join(filepath_images,filenames[i]))
            writer.append_data(image)
    print('Animation has been generated (using %d frames per second) and saved at %s'%(fps,fn_out))
    
###################################################################################################
# VALIDATION
###################################################################################################

def compare_timeseries(ts,gt,key,settings):
    if key not in gt.keys():
        raise Exception('transect name %s does not exist in grountruth file'%key)
    # remove nans
    chainage = np.array(ts[key])
    idx_nan = np.isnan(chainage)
    dates_nonans = [ts['dates'][k].to_pydatetime() for k in np.where(~idx_nan)[0]]
    satnames_nonans = [ts['satname'][k] for k in np.where(~idx_nan)[0]]
    chain_nonans = chainage[~idx_nan]
    # define satellite and survey time-series
    chain_sat_dm = chain_nonans
    chain_sur_dm = gt[key]['chainages']
    # plot the time-series
    fig= plt.figure(figsize=[15,8], tight_layout=True)
    gs = gridspec.GridSpec(2,3)
    ax0 = fig.add_subplot(gs[0,:])
    ax0.grid(which='major',linestyle=':',color='0.5')
    ax0.plot(gt[key]['dates'], chain_sur_dm,'-o',mfc='w',ms=3,label='in situ')
    ax0.plot(dates_nonans, chain_sat_dm,'-o',mfc='w',ms=3,label='satellite')
    ax0.set(title= 'Transect ' + key, xlim=[dates_nonans[0]-timedelta(days=30),
                                            dates_nonans[-1]+timedelta(days=30)])#,ylim=sett['lims'])
    ax0.legend(loc='upper left')
    
    # interpolate surveyed data around satellite data based on the parameters (min_days and max_days)
    chain_int = np.nan*np.ones(len(dates_nonans))
    for k,date in enumerate(dates_nonans):
        # compute the days distance for each satellite date
        days_diff = np.array([ (_ - date).days for _ in gt[key]['dates']])
        # if nothing within max_days put a nan
        if np.min(np.abs(days_diff)) > settings['max_days']:
            chain_int[k] = np.nan
        else:
            # if a point within min_days, take that point (no interpolation)
            if np.min(np.abs(days_diff)) < settings['min_days']:
                idx_closest = np.where(np.abs(days_diff) == np.min(np.abs(days_diff)))
                chain_int[k] = float(gt[key]['chainages'][idx_closest[0][0]])
            else: # otherwise, between min_days and max_days, interpolate between the 2 closest points
                if sum(days_diff > 0) == 0:
                    break
                idx_after = np.where(days_diff > 0)[0][0]
                idx_before = idx_after - 1
                x = [gt[key]['dates'][idx_before].toordinal() , gt[key]['dates'][idx_after].toordinal()]
                y = [gt[key]['chainages'][idx_before], gt[key]['chainages'][idx_after]]
                f = interpolate.interp1d(x, y,bounds_error=True)
                try:
                    chain_int[k] = float(f(date.toordinal()))
                except:
                    chain_int[k] = np.nan
    # remove nans again
    idx_nan = np.isnan(chain_int)
    chain_sat = chain_nonans[~idx_nan]
    chain_sur = chain_int[~idx_nan]
    dates_sat = [dates_nonans[k] for k in np.where(~idx_nan)[0]]
    satnames = [satnames_nonans[k] for k in np.where(~idx_nan)[0]]
    if len(chain_sat) < 8 or len(chain_sur) < 8: 
        return  chain_sat, chain_sur, satnames, fig
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
        ax2.text(j+1+0.35,median_data[j]+1, ('n=%.d' % int(n_data[j])), ha='center', va='center', fontsize=12,
                 rotation='vertical')
    ax2.set(ylabel='error [m]', ylim=settings['lims'])
    
    # histogram
    ax3 = fig.add_subplot(gs[1,2])
    ax3.grid(which='major',linestyle=':',color='0.5')
    ax3.axvline(x=0, ls='--', lw=1.5, color='k')
    binwidth = settings['binwidth']
    bins = np.arange(min(chain_error), max(chain_error) + binwidth, binwidth)
    density = plt.hist(chain_error, bins=bins, density=True, color='0.6', edgecolor='k', alpha=0.5)
    mu, std = stats.norm.fit(chain_error)
    pval = stats.normaltest(chain_error)[1]
    xlims = ax3.get_xlim()
    x = np.linspace(xlims[0], xlims[1], 100)
    p = stats.norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r-', linewidth=1)
    ax3.set(xlabel='error [m]', ylabel='pdf', xlim=settings['lims'])
    str_stats = ' rmse = %.1f\n mean = %.1f\n std = %.1f\n q90 = %.1f' % (rmse, mean, std, q90)
    ax3.text(0, 0.98, str_stats,va='top', transform=ax3.transAxes)    
    
    return chain_sat, chain_sur, satnames, fig
