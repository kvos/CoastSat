"""
This module contains all the functions needed to preprocess the satellite images
before the shorelines can be extracted. This includes creating a cloud mask and
pansharpening/downsampling the multispectral bands.

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
from skimage.io import imsave
from skimage import img_as_ubyte

# other modules
from osgeo import gdal
from pyproj import CRS
from pylab import ginput
import pickle
import geopandas as gpd
import pandas as pd
from shapely import geometry
import re

# CoastSat modules
from coastsat import SDS_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

# Main function to preprocess a satellite image (L5, L7, L8, L9 or S2)
def preprocess_single(fn, satname, cloud_mask_issue, pan_off, collection, s2cloudless_prob=40):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image.
    For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it
    also outputs the 20m SWIR band.

    KV WRL 2018

    Arguments:
    -----------
    fn: str or list of str
        filename of the .TIF file containing the image. For L7, L8 and S2 this
        is a list of filenames, one filename for each band at different
        resolution (30m and 15m for Landsat 7-8, 10m, 20m, 60m for Sentinel-2)
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being masked on the images
    pan_off : boolean
        if True, disable panchromatic sharpening and ignore pan band
    collection: str
        Landsat collection ,'C01' or 'C02'
    s2cloudless_prob: float [0,100)
        threshold to identify cloud pixels in the s2cloudless probability mask
        
    Returns:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the
        coordinates of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located

    """
    
    if isinstance(fn, list):
        fn_to_split=fn[0]
    elif isinstance(fn, str):
        fn_to_split=fn
    # split by os.sep and only get the filename at the end then split again to remove file extension
    fn_to_split=fn_to_split.split(os.sep)[-1].split('.')[0]
    # search for the year the tif was taken with regex and convert to int
    year = int(re.search('[0-9]+',fn_to_split).group(0))
    # after 2022 everything is automatically from Collection 2
    if collection == 'C01' and year >= 2022:
        collection = 'C02'
        
    #=============================================================================================#
    # L5 images
    #=============================================================================================#
    if satname == 'L5':
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_mask = fn[1]
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)
        # read cloud mask
        data = gdal.Open(fn_mask, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_QA = bands[0]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)

        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []

    #=============================================================================================#
    # L7, L8 and L9 images
    #=============================================================================================#
    elif satname in ['L7','L8','L9']:
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_pan = fn[1]  
        fn_mask = fn[2]  
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)
        # read cloud mask
        data = gdal.Open(fn_mask, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_QA = bands[0]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata) 
        
        # if panchromatic sharpening is turned off
        if pan_off:            
            # ms bands are untouched and the extra image is empty
            im_extra = []
    
        # otherwise perform panchromatic sharpening
        else:
            # read panchromatic band
            data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
            im_pan = bands[0]
           
            # pansharpen Green, Blue, NIR for Landsat 7
            if satname == 'L7':
                try:
                    im_ms_ps = pansharpen(im_ms[:,:,[1,2,3]], im_pan, cloud_mask)
                except: # if pansharpening fails, keep downsampled bands (for long runs)
                    print('\npansharpening of image %s failed.'%fn[0])
                    im_ms_ps = im_ms[:,:,[1,2,3]]
                # add downsampled Blue and SWIR1 bands
                im_ms_ps = np.append(im_ms[:,:,[0]], im_ms_ps, axis=2)
                im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[4]], axis=2)
                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan
                
            # pansharpen Blue, Green, Red for Landsat 8 and 9           
            elif satname in ['L8','L9']:
                try:
                    im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
                except: # if pansharpening fails, keep downsampled bands (for long runs)
                    print('\npansharpening of image %s failed.'%fn[0])
                    im_ms_ps = im_ms[:,:,[0,1,2]]
                # add downsampled NIR and SWIR1 bands
                im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)
                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan
                
    #=============================================================================================#
    # S2 images
    #=============================================================================================#
    if satname == 'S2':
        # read 10m bands (R,G,B,NIR)
        fn_ms = fn[0]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount-1)]
        im_ms = np.stack(bands, 2)
        im_ms = im_ms/10000 # TOA scaled to 10000
        # read s2cloudless cloud probability (last band in ms image)
        cloud_prob = data.GetRasterBand(data.RasterCount).ReadAsArray()

        # image size
        nrows = im_ms.shape[0]
        ncols = im_ms.shape[1]
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im_ms))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((nrows,ncols)).astype('bool')
            return im_ms, georef, cloud_mask, [], [], []

        # read 20m band (SWIR1)
        fn_swir = fn[1]
        data = gdal.Open(fn_swir, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_swir = bands[0]
        im_swir = im_swir/10000 # TOA scaled to 10000
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im_ms, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn_mask = fn[2]
        data = gdal.Open(fn_mask, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_QA = bands[0]
        # compute cloud mask using QA60 band
        cloud_mask_QA60 = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
        # compute cloud mask using s2cloudless probability band
        cloud_mask_s2cloudless = create_s2cloudless_mask(cloud_prob, s2cloudless_prob)
        # combine both cloud masks
        cloud_mask = np.logical_or(cloud_mask_QA60,cloud_mask_s2cloudless)
        
        # check if -inf or nan values on any band and create nodata image
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # add the edges of the SWIR1 band that contains only 0's to the nodata image
        # these are created when reprojecting the SWIR1 20 m band onto the 10m pixel grid
        im_nodata = pad_edges(im_swir, im_nodata)        
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(im_nodata.shape).astype(bool)
        im_zeros = np.logical_and(np.isin(im_ms[:,:,1],0), im_zeros) # Green
        im_zeros = np.logical_and(np.isin(im_ms[:,:,3],0), im_zeros) # NIR
        im_zeros = np.logical_and(np.isin(im_ms[:,:,4],0), im_zeros) # SWIR
        # add to im_nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # dilate if image was merged as there could be issues at the edges
        if 'merged' in fn_ms:
            im_nodata = morphology.dilation(im_nodata,morphology.square(5))

        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image
        im_extra = []

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata

###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def find_edge_padding(im_band: np.ndarray) -> np.ndarray:
    """
    Finds the padding required for each edge of an image band based on the presence of data.

    Parameters:
    im_band (numpy.ndarray): The image band.

    Returns:
    tuple: A tuple containing the top, bottom, left, and right padding values.
    """
    # Assuming non-data values are zeros. Adjust the condition if needed.
    is_data = im_band != 0

    # Function to find padding for one edge
    def find_edge_data(is_data_along_edge):
        for idx, has_data in enumerate(is_data_along_edge):
            if has_data:
                return idx
        return len(is_data_along_edge)  # Return full length if no data found

    # Calculate padding for each side
    top_padding = find_edge_data(np.any(is_data, axis=1))
    bottom_padding = find_edge_data(np.any(is_data, axis=1)[::-1])
    left_padding = find_edge_data(np.any(is_data, axis=0))
    right_padding = find_edge_data(np.any(is_data, axis=0)[::-1])

    return top_padding, bottom_padding, left_padding, right_padding


def pad_edges(im_swir: np.ndarray, im_nodata: np.ndarray) -> np.ndarray:
    """
    Adds 0's located along the edges of im_swir to the nodata array.

    Fixes the issue where 0s are created along the edges of the SWIR1 band caused by reprojecting the 20 m band onto the 10m pixel grid (with bilinear interpolation in GDAL)

    Args:
        im_swir (np.ndarray): The SWIR image.
        im_nodata (np.ndarray): The nodata array.

    Returns:
        np.ndarray: The nodata array with padded edges.
    """
    top_pad, bottom_pad, left_pad, right_pad = find_edge_padding(im_swir)
    # Apply this padding to your masks or other arrays as needed

    # if bottom pad is 0 the entire image gets set to True
    if bottom_pad > 0:
        im_nodata[-bottom_pad:, :] = True
    # if right pad is 0 the entire image gets set to True
    if right_pad > 0:
        im_nodata[:, -right_pad:] = True

    im_nodata[:, :left_pad] = True
    im_nodata[:top_pad, :] = True
    return im_nodata


def create_cloud_mask(im_QA, satname, cloud_mask_issue, collection):
    """
    Creates a cloud mask using the information contained in the QA band.

    KV WRL 2018

    Arguments:
    -----------
    im_QA: np.array
        Image containing the QA band
    satname: string
        short name for the satellite: ```'L5', 'L7', 'L8' or 'S2'```
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being
        erroneously masked on the images
    collection: str
        Landsat collection ,'C01' or 'C02'
        
    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise

    """
    if satname == 'S2':
        # 1024 = dense cloud, 2048 = cirrus clouds
        cloud_values = [1024, 2048] 
    else:
        if collection == 'C01':
            if  satname in ['L8','L9']:
                # 2800, 2804, 2808, 2812 = High confidence cloud
                # 6896, 6900, 6904, 6908 = High confidence cirrus cloud
                cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
            elif satname in ['L4','L5','L7','L8']:
                # 752, 756, 760, 764 = High confidence cloud
                cloud_values = [752, 756, 760, 764]
        elif collection == 'C02':
            # function to return flag for n-th bit
            def is_set(x, n):
                return x & 1 << n != 0   
            # dilated cloud = bit 1
            # cirrus = bit 2
            # cloud = bit 3 
            qa_values = np.unique(im_QA.flatten())
            cloud_values = []
            for qaval in qa_values:
                for k in [1,2,3]: # check the first 3 flags
                    if is_set(qaval,k):
                        cloud_values.append(qaval)
 
    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_QA, cloud_values)

    # remove cloud pixels that form very thin features. These are beach or swash pixels that are
    # erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        cloud_mask = morphology.remove_small_objects(cloud_mask, min_size=40, connectivity=1)

        if cloud_mask_issue:
            elem = morphology.square(6) # use a square of width 6 pixels
            cloud_mask = morphology.binary_opening(cloud_mask,elem) # perform image opening
            # remove objects with less than min_size connected pixels
            cloud_mask = morphology.remove_small_objects(cloud_mask, min_size=100, connectivity=1)

    return cloud_mask

def create_s2cloudless_mask(cloud_prob, s2cloudless_prob):
    """
    Creates a cloud mask using the s2cloudless band.

    KV WRL 2023

    Arguments:
    -----------
    cloud_prob: np.array
        Image containing the s2cloudless cloud probability
        
    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise

    """
    # find which pixels have bits corresponding to cloud values
    cloud_mask = cloud_prob > s2cloudless_prob
    # dilate cloud mask
    elem = morphology.square(6) # use a square of width 6 pixels
    cloud_mask = morphology.binary_opening(cloud_mask,elem) # perform image opening

    return cloud_mask

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches
    that of a target image.

    Arguments:
    -----------
    source: np.array
        Image to transform; the histogram is computed over the flattened
        array
    template: np.array
        Template image; can have different dimensions to source

    Returns:
    -----------
    matched: np.array
        The transformed output image

    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def pansharpen(im_ms, im_pan, cloud_mask):
    """
    Pansharpens a multispectral image, using the panchromatic band and a cloud mask.
    A PCA is applied to the image, then the 1st PC is replaced, after histogram
    matching with the panchromatic band. Note that it is essential to match the
    histrograms of the 1st PC and the panchromatic band before replacing and
    inverting the PCA.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Multispectral image to pansharpen (3D)
    im_pan: np.array
        Panchromatic band (2D)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:
    -----------
    im_ms_ps: np.ndarray
        Pansharpened multispectral image (3D)

    """
    # check that cloud cover is not too high otherwise pansharpening fails
    if sum(sum(cloud_mask)) > 0.95*cloud_mask.shape[0]*cloud_mask.shape[1]:
        return im_ms
    
    # reshape image into vector and apply cloud mask
    vec = im_ms.reshape(im_ms.shape[0] * im_ms.shape[1], im_ms.shape[2])
    vec_mask = cloud_mask.reshape(im_ms.shape[0] * im_ms.shape[1])
    vec = vec[~vec_mask, :]
    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pcs = pca.fit_transform(vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_pan = im_pan.reshape(im_pan.shape[0] * im_pan.shape[1])
    vec_pan = vec_pan[~vec_mask]
    vec_pcs[:,0] = hist_match(vec_pan, vec_pcs[:,0])
    vec_ms_ps = pca.inverse_transform(vec_pcs)

    # reshape vector into image
    vec_ms_ps_full = np.ones((len(vec_mask), im_ms.shape[2])) * np.nan
    vec_ms_ps_full[~vec_mask,:] = vec_ms_ps
    im_ms_ps = vec_ms_ps_full.reshape(im_ms.shape[0], im_ms.shape[1], im_ms.shape[2])

    return im_ms_ps

def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high))
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj

def create_jpg(im_ms, cloud_mask, date, satname, filepath, use_matplotlib=True):
    """
    Saves a .jpg file with the RGB image as well as the NIR and SWIR1 grayscale images.
    This functions can be modified to obtain different visualisations of the
    multispectral images.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    date: str
        string containing the date at which the image was acquired
    satname: str
        name of the satellite mission (e.g., 'L5')
    filepath: str
        directory in which to save the images
    use_matplotlib: boolean
        False to save a .jpg and True to save as matplotlib plots

    Returns:
    -----------
        Saves a .jpg image corresponding to the preprocessed satellite image

    """
    # rescale image intensity for display purposes
    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_NIR = rescale_image_intensity(im_ms[:,:,3], cloud_mask, 99.9)
    im_SWIR = rescale_image_intensity(im_ms[:,:,4], cloud_mask, 99.9)
    
    # creates raw jpg files that can be used for ML applications
    if not use_matplotlib:
        # convert images to bytes so they can be saved
        im_RGB = img_as_ubyte(im_RGB)
        im_NIR = img_as_ubyte(im_NIR)
        im_SWIR = img_as_ubyte(im_SWIR)
        # Save each kind of image with skimage.io
        file_types = ["RGB","SWIR","NIR"]
        # create folders RGB, SWIR, and NIR to hold each type of image
        for ext in file_types:
            ext_filepath = filepath + os.sep + ext
            if not os.path.exists(ext_filepath):
                os.mkdir(ext_filepath)
            # location to save image rgb image would be in sitename/RGB/sitename.jpg
            fname=os.path.join(ext_filepath, date + '_'+ ext +'_' + satname + '.jpg')
            if ext == "RGB":
                imsave(fname, im_RGB, quality=100)
            if ext == "SWIR":
                imsave(fname, im_SWIR, quality=100)
            if ext == "NIR":
                imsave(fname, im_NIR, quality=100)
                
    # if use_matplotlib=True, creates a nicer plot
    else:
        fig = plt.figure()
        fig.set_size_inches([18,9])
        fig.set_tight_layout(True)
        ax1 = fig.add_subplot(111)
        ax1.axis('off')
        ax1.imshow(im_RGB)
        ax1.set_title(date + '   ' + satname, fontsize=16)
        
        # choose vertical or horizontal based on image size
        # if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
        #     ax1 = fig.add_subplot(311)
        #     ax2 = fig.add_subplot(312)
        #     ax3 = fig.add_subplot(313)
        # else:
        #     ax1 = fig.add_subplot(131)
        #     ax2 = fig.add_subplot(132)
        #     ax3 = fig.add_subplot(133)
        # # RGB
        # ax1.axis('off')
        # ax1.imshow(im_RGB)
        # ax1.set_title(date + '   ' + satname, fontsize=16)
        # # NIR
        # ax2.axis('off')
        # ax2.imshow(im_NIR, cmap='seismic')
        # ax2.set_title('Near Infrared', fontsize=16)
        # # SWIR
        # ax3.axis('off')
        # ax3.imshow(im_SWIR, cmap='seismic')
        # ax3.set_title('Short-wave Infrared', fontsize=16)
    
        # save figure
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

def save_jpg(metadata, settings, use_matplotlib=False):
    """
    Saves a .jpg image for all the images contained in metadata.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        's2cloudless_prob': float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask
        'use_matplotlib': boolean
            False to save a .jpg and True to save as matplotlib plots

    Returns:
    -----------
    Stores the images as .jpg in a folder named /preprocessed

    """

    sitename = settings['inputs']['sitename']
    cloud_thresh = settings['cloud_thresh']
    s2cloudless_prob = settings['s2cloudless_prob']
    filepath_data = settings['inputs']['filepath']
    collection = settings['inputs']['landsat_collection']
    
    # create subfolder to store the jpg files
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'preprocessed')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)

    # loop through satellite list
    print('Saving images as jpg:')
    for satname in metadata.keys():
        
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        print('%s: %d images'%(satname,len(filenames)))
        # loop through images
        for i in range(len(filenames)):
            print('\r%d%%' %int((i+1)/len(filenames)*100), end='')
            # image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(fn, satname, settings['cloud_mask_issue'],
                                                                                      settings['pan_off'], collection, 
                                                                                      s2cloudless_prob)

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
                continue

            # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            # skip image if cloud cover is above threshold
            if cloud_cover > cloud_thresh or cloud_cover == 1:
                continue
            # save .jpg with date and satellite in the title
            date = filenames[i][:19]
            plt.ioff()  # turning interactive plotting off
            create_jpg(im_ms, cloud_mask, date, satname, filepath_jpg, use_matplotlib)
        print('')
    # print the location where the images have been saved
    print('Satellite images saved as .jpg in ' + os.path.join(filepath_data, sitename,
                                                    'jpg_files', 'preprocessed'))

def get_reference_sl(metadata, settings):
    """
    Allows the user to manually digitize a reference shoreline that is used seed
    the shoreline detection algorithm. The reference shoreline helps to detect
    the outliers, making the shoreline detection more robust.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'output_epsg': int
            output spatial reference system as EPSG code

    Returns:
    -----------
    reference_shoreline: np.array
        coordinates of the reference shoreline that was manually digitized.
        This is also saved as a .pkl and .geojson file.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    collection = settings['inputs']['landsat_collection']
    pts_coords = []
    # check if reference shoreline already exists in the corresponding folder
    fp_ref_shoreline = os.path.join(filepath_data, sitename, sitename + '_reference_shoreline.geojson')
    # if it exist, load it and load the geojson file
    if os.path.exists(fp_ref_shoreline):
        print('Reference shoreline already exists and was loaded')
        refsl_geojson = gpd.read_file(fp_ref_shoreline,driver='GeoJSON')
        refsl = np.array(refsl_geojson.iloc[0]['geometry'].coords)
        print('Reference shoreline coordinates are in epsg:%d'%refsl_geojson.crs.to_epsg())
        return refsl

    # otherwise get the user to manually digitise a shoreline on 
    # S2, L8, L9 or L5 images (no L7 because of scan line error)
    # first try to use S2 images (10m res for manually digitizing the reference shoreline)
    if 'S2' in metadata.keys(): satname = 'S2'
    # if no S2 images, use L8 or L9 (15m res in the RGB with pansharpening)
    elif 'L8' in metadata.keys(): satname = 'L8'
    elif 'L9' in metadata.keys(): satname = 'L9'
    # if no S2, L8 or L9 use L5 (30m res)
    elif 'L5' in metadata.keys(): satname = 'L5'
    # if only L7 images, ask user to download other images
    else:
        raise Exception('You cannot digitize the shoreline on L7 images (because of gaps in the images), add another L8, S2 or L5 to your dataset.')
    filepath = SDS_tools.get_filepath(settings['inputs'],satname)
    filenames = metadata[satname]['filenames']
    # create figure
    fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # loop trhough the images
    for i in range(len(filenames)):

        # read image
        fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(fn, satname, settings['cloud_mask_issue'],
                                                                                  settings['pan_off'], collection,
                                                                                  settings['s2cloudless_prob'])

        # compute cloud_cover percentage (with no data pixels)
        cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                (cloud_mask.shape[0]*cloud_mask.shape[1]))
        if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
            continue

        # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
        cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
        # compute updated cloud cover percentage (without no data pixels)
        cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                (sum(sum((~im_nodata).astype(int)))))

        # skip image if cloud cover is above threshold
        if cloud_cover > settings['cloud_thresh']:
            continue

        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

        # plot the image RGB on a figure
        ax.axis('off')
        ax.imshow(im_RGB)

        # decide if the image if good enough for digitizing the shoreline
        ax.set_title('Press <right arrow> if image is clear enough to digitize the shoreline.\n' +
                  'If the image is cloudy press <left arrow> to get another image', fontsize=14)
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        skip_image = False
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

        if skip_image:
            ax.clear()
            continue
        else:
            # create two new buttons
            add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            # add multiple reference shorelines (until user clicks on <end> button)
            pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
            geoms = []
            while 1:
                add_button.set_visible(False)
                end_button.set_visible(False)
                # update title (instructions)
                ax.set_title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                          'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                          fontsize=14)
                plt.draw()

                # let user click on the shoreline
                pts = ginput(n=50000, timeout=-1, show_clicks=True)
                pts_pix = np.array(pts)
                # convert pixel coordinates to world coordinates
                pts_world = SDS_tools.convert_pix2world(pts_pix[:,[1,0]], georef)

                # interpolate between points clicked by the user (1m resolution)
                pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                for k in range(len(pts_world)-1):
                    pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                    xvals = np.arange(0,pt_dist)
                    yvals = np.zeros(len(xvals))
                    pt_coords = np.zeros((len(xvals),2))
                    pt_coords[:,0] = xvals
                    pt_coords[:,1] = yvals
                    phi = 0
                    deltax = pts_world[k+1,0] - pts_world[k,0]
                    deltay = pts_world[k+1,1] - pts_world[k,1]
                    phi = np.pi/2 - np.math.atan2(deltax, deltay)
                    tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                    pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
                pts_world_interp = np.delete(pts_world_interp,0,axis=0)

                # save as geometry (to create .geojson file later)
                geoms.append(geometry.LineString(pts_world_interp))

                # convert to pixel coordinates and plot
                pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
                pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

                # update title and buttons
                add_button.set_visible(True)
                end_button.set_visible(True)
                ax.set_title('click on <add> to digitize another shoreline or on <end> to finish and save the shoreline(s)',
                          fontsize=14)
                plt.draw()

                # let the user click again (<add> another shoreline or <end>)
                pt_input = ginput(n=1, timeout=-1, show_clicks=False)
                pt_input = np.array(pt_input)

                # if user clicks on <end>, save the points and break the loop
                if pt_input[0][0] > im_ms.shape[1]/2:
                    add_button.set_visible(False)
                    end_button.set_visible(False)
                    plt.title('Reference shoreline saved as ' + sitename + '_reference_shoreline.pkl and ' + sitename + '_reference_shoreline.geojson')
                    plt.draw()
                    ginput(n=1, timeout=3, show_clicks=False)
                    plt.close()
                    break

            pts_sl = np.delete(pts_sl,0,axis=0)
            # convert world image coordinates to user-defined coordinate system
            image_epsg = metadata[satname]['epsg'][i]
            pts_coords = SDS_tools.convert_epsg(pts_sl, image_epsg, settings['output_epsg'])

            # save the reference shoreline as .pkl
            filepath = os.path.join(filepath_data, sitename)
            with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'wb') as f:
                pickle.dump(pts_coords, f)

            # also store as .geojson in case user wants to drag-and-drop on GIS for verification
            for k,line in enumerate(geoms):
                gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(line))
                gdf.index = [k]
                gdf.loc[k,'name'] = 'reference shoreline ' + str(k+1)
                # store into geodataframe
                if k == 0:
                    gdf_all = gdf
                else:
                    gdf_all = pd.concat([gdf_all, gdf])
            gdf_all.crs = CRS(image_epsg)
            # convert from image_epsg to user-defined coordinate system
            gdf_all = gdf_all.to_crs(epsg=settings['output_epsg'])
            # save as geojson
            gdf_all.to_file(os.path.join(filepath, sitename + '_reference_shoreline.geojson'),
                            driver='GeoJSON', encoding='utf-8')

            print('Reference shoreline has been saved in ' + filepath)
            break

    # check if a shoreline was digitised
    if len(pts_coords) == 0:
        raise Exception('No cloud free images are available to digitise the reference shoreline,'+
                        'download more images and try again')
    
    return pts_coords
