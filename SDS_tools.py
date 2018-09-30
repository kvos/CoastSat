"""This module contains utilities to work with satellite images' 
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# Initial settings
import os
import numpy as np
from osgeo import gdal, ogr, osr
import skimage.transform as transform
import simplekml
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
    
def get_filenames(filename, filepath, satname):
    
    if satname == 'L5':
        fn = os.path.join(filepath, filename)
    if satname == 'L7' or satname == 'L8':
        idx = filename.find('.tif')
        filename_ms = filename[:idx-3] + 'ms.tif'
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        idx = filename.find('.tif')
        filename20 = filename[:idx-3] + '20m.tif'
        filename60 = filename[:idx-3] + '60m.tif'
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename20),
              os.path.join(filepath[2], filename60)]
    return fn
    
        
    