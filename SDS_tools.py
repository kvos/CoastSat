"""This module contains utilities to work with satellite images' 
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# Initial settings
import os
import numpy as np
from osgeo import gdal, ogr, osr
import skimage.transform as transform
import simplekml
from scipy.ndimage.filters import uniform_filter
import pdb

# Functions

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (row,columns) to world projected coordinates
    performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
        points: np.array or list of np.array
            array with 2 columns (rows first and columns second)
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    -----------
        points_converted: np.array or list of np.array 
            converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
            
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates (row,column)
    performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
        points: np.array or list of np.array
            array with 2 columns (rows first and columns second)
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    -----------
        points_converted: np.array or list of np.array 
            converted coordinates, first columns with row and second column with column
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes.
    
    KV WRL 2018

    Arguments:
    -----------
        points: np.array or list of np.ndarray
            array with 2 columns (rows first and columns second)
        epsg_in: int
            epsg code of the spatial reference in which the input is
        epsg_out: int
            epsg code of the spatial reference in which the output will be            
                
    Returns:    -----------
        points_converted: np.array or list of np.array 
            converted coordinates
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # transform points
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        print('invalid input type')
        raise
        
    return points_converted

def coords_from_kml(fn):
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def save_kml(coords, epsg):
    
    kml = simplekml.Kml()
    coords_wgs84 = convert_epsg(coords, epsg, 4326)
    kml.newlinestring(name='coords', coords=coords_wgs84)
    kml.save('coords.kml')
    
def get_filepath(inputs,satname):
    
    sitename = inputs['sitename']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        filepath = os.path.join(os.getcwd(), 'data', sitename, satname, '30m')
    elif satname == 'L7':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(os.getcwd(), 'data', sitename, 'L7', 'pan')
        filepath_ms = os.path.join(os.getcwd(), 'data', sitename, 'L7', 'ms')
        filenames_pan = os.listdir(filepath_pan)
        filenames_ms = os.listdir(filepath_ms)
        if (not len(filenames_pan) == len(filenames_ms)):
            raise 'error: not the same amount of files for pan and ms'
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'L8':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(os.getcwd(), 'data', sitename, 'L8', 'pan')
        filepath_ms = os.path.join(os.getcwd(), 'data', sitename, 'L8', 'ms')
        filenames_pan = os.listdir(filepath_pan)
        filenames_ms = os.listdir(filepath_ms)
        if (not len(filenames_pan) == len(filenames_ms)):
            raise 'error: not the same amount of files for pan and ms'
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        filepath10 = os.path.join(os.getcwd(), 'data', sitename, satname, '10m')
        filenames10 = os.listdir(filepath10)
        filepath20 = os.path.join(os.getcwd(), 'data', sitename, satname, '20m')
        filenames20 = os.listdir(filepath20)
        filepath60 = os.path.join(os.getcwd(), 'data', sitename, satname, '60m')
        filenames60 = os.listdir(filepath60)
        if (not len(filenames10) == len(filenames20)) or (not len(filenames20) == len(filenames60)):
            raise 'error: not the same amount of files for 10, 20 and 60 m bands'
        filepath = [filepath10, filepath20, filepath60]
            
    return filepath
    
def get_filenames(filename, filepath, satname):
    
    if satname == 'L5':
        fn = os.path.join(filepath, filename)
    if satname == 'L7' or satname == 'L8':
        filename_ms = filename.replace('pan','ms')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        filename20 = filename.replace('10m','20m')
        filename60 = filename.replace('10m','60m')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename20),
              os.path.join(filepath[2], filename60)]
        
    return fn
    
def image_std(image, radius):
    
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size is always 3x3
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std
    win_mean = uniform_filter(image_padded, (win_rows, win_cols))
    win_sqr_mean = uniform_filter(image_padded**2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std

def mask_raster(fn, mask):
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
    
         
    