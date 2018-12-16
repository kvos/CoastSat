"""This module contains all the functions needed to preprocess the satellite images before the
shoreline can be extracted. This includes creating a cloud mask and 
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

# other modules
from osgeo import gdal, ogr, osr
from pylab import ginput
import pickle
import matplotlib.path as mpltPath

# own modules
import SDS_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

def create_cloud_mask(im_qa, satname):
    """
    Creates a cloud mask using the information contained in the QA band.
    
    KV WRL 2018
    
    Arguments:
    -----------
        im_qa: np.array
            Image containing the QA band
        satname: string
            short name for the satellite (L5, L7, L8 or S2)
            
    Returns:
    -----------
        cloud_mask : np.array
            A boolean array with True if a pixel is cloudy and False otherwise
    """
    
    # convert QA bits (the bits allocated to cloud cover vary depending on the satellite mission)
    if satname == 'L8':
        cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
    elif satname == 'L7' or satname == 'L5' or satname == 'L4':
        cloud_values = [752, 756, 760, 764]
    elif satname == 'S2':
        cloud_values = [1024, 2048] # 1024 = dense cloud, 2048 = cirrus clouds
    
    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_qa, cloud_values)
    
    # remove isolated cloud pixels (there are some in the swash zone and they are not clouds)
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        morphology.remove_small_objects(cloud_mask, min_size=10, connectivity=1, in_place=True)
    
    return cloud_mask

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches that of a 
    target image.

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
    A PCA is applied to the image, then the 1st PC is replaced with the panchromatic band.
    Note that it is essential to match the histrograms of the 1st PC and the panchromatic band
    before replacing and inverting the PCA.
    
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
            The rescaled image
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

def preprocess_single(fn, satname):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands, the 
    georeferencing vector of the image (coordinates of the upper left pixel), the cloud mask and 
    the QA band. For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it also 
    outputs the 20m SWIR band.
    
    KV WRL 2018

    Arguments:
    -----------
        fn: str or list of str
            filename of the .TIF file containing the image
            for L7, L8 and S2 this is a list of filenames, one filename for each band at different
            resolution (30m and 15m for Landsat 7-8, 10m, 20m, 60m for Sentinel-2)
        satname: str
            name of the satellite mission (e.g., 'L5')
        
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
        imQA: np.array
            2D array containing the QA band, from which the cloud_mask can be computed.
            
    """
        
    #=============================================================================================#
    # L5 images
    #=============================================================================================#
    if satname == 'L5':
        
        # read all bands
        data = gdal.Open(fn, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)
        
        # down-sample to 15 m (half of the original pixel size)
        nrows = im_ms.shape[0]*2
        ncols = im_ms.shape[1]*2
        
        # create cloud mask
        im_qa = im_ms[:,:,5]
        im_ms = im_ms[:,:,:-1]
        cloud_mask = create_cloud_mask(im_qa, satname)

        # resize the image using bilinear interpolation (order 1)
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        
        # adjust georeferencing vector to the new image size
        # scale becomes 15m and the origin is adjusted to the center of new top left pixel
        georef[1] = 15
        georef[5] = -15
        georef[0] = georef[0] + 7.5
        georef[3] = georef[3] - 7.5
        
        # check if -inf or nan values on any band and add to cloud mask
        for k in range(im_ms.shape[2]):   
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            
        # calculate cloud cover
        cloud_cover = sum(sum(cloud_mask.astype(int)))/(cloud_mask.shape[0]*cloud_mask.shape[1])
        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []
        imQA = im_qa
        
    #=============================================================================================#
    # L7 images
    #=============================================================================================#               
    elif satname == 'L7':
        
        # read pan image
        fn_pan = fn[0]
        data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_pan = np.stack(bands, 2)[:,:,0]
        
        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]
        
        # read ms image
        fn_ms = fn[1]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)
        
        # create cloud mask
        im_qa = im_ms[:,:,5]
        cloud_mask = create_cloud_mask(im_qa, satname)
        
        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_') 
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        for k in range(im_ms.shape[2]+1): 
            if k == 5:
                im_inf = np.isin(im_pan, -np.inf)
                im_nan = np.isnan(im_pan)        
            else:  
                im_inf = np.isin(im_ms[:,:,k], -np.inf)
                im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            
        # calculate cloud cover
        cloud_cover = sum(sum(cloud_mask.astype(int)))/(cloud_mask.shape[0]*cloud_mask.shape[1])
        
        # pansharpen Green, Red, NIR (where there is overlapping with pan band in L7)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[1,2,3]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[1,2,3]]
        # add downsampled Blue and SWIR1 bands
        im_ms_ps = np.append(im_ms[:,:,[0]], im_ms_ps, axis=2)
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[4]], axis=2)
        
        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan
        imQA = im_qa
        
    #=============================================================================================#
    # L8 images
    #=============================================================================================#               
    elif satname == 'L8':
        
        # read pan image
        fn_pan = fn[0]
        data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_pan = np.stack(bands, 2)[:,:,0]
        
        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]
        
        # read ms image
        fn_ms = fn[1]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)
        
        # create cloud mask
        im_qa = im_ms[:,:,5]
        cloud_mask = create_cloud_mask(im_qa, satname)
        
        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_') 
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        for k in range(im_ms.shape[2]+1): 
            if k == 5:
                im_inf = np.isin(im_pan, -np.inf)
                im_nan = np.isnan(im_pan)        
            else:  
                im_inf = np.isin(im_ms[:,:,k], -np.inf)
                im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            
        # calculate cloud cover
        cloud_cover = sum(sum(cloud_mask.astype(int)))/(cloud_mask.shape[0]*cloud_mask.shape[1])

        # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[0,1,2]]
        # add downsampled NIR and SWIR1 bands
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)
        
        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan
        imQA = im_qa   
        
    #=============================================================================================#
    # S2 images
    #=============================================================================================#               
    if satname == 'S2':
        
        # read 10m bands (R,G,B,NIR)
        fn10 = fn[0]
        data = gdal.Open(fn10, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im10 = np.stack(bands, 2)
        im10 = im10/10000 # TOA scaled to 10000
        
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im10))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((im10.shape[0],im10.shape[1])).astype('bool')
            return im_ms, georef, cloud_mask, [], []
        
        # size of 10m bands
        nrows = im10.shape[0]
        ncols = im10.shape[1]
        
        # read 20m band (SWIR1)
        fn20 = fn[1]
        data = gdal.Open(fn20, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im20 = np.stack(bands, 2)
        im20 = im20[:,:,0]
        im20 = im20/10000 # TOA scaled to 10000
        
        # resize the image using bilinear interpolation (order 1)
        im_swir = transform.resize(im20, (nrows, ncols), order=1, preserve_range=True,
                                   mode='constant')
        im_swir = np.expand_dims(im_swir, axis=2)
        
        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im10, im_swir, axis=2)
        
        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn60 = fn[2]
        data = gdal.Open(fn60, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im60 = np.stack(bands, 2)
        imQA = im60[:,:,0]
        cloud_mask = create_cloud_mask(imQA, satname)
        # resize the cloud mask using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask,(nrows, ncols), order=0, preserve_range=True,
                                      mode='constant')
        # check if -inf or nan values on any band and add to cloud mask
        for k in range(im_ms.shape[2]):   
                im_inf = np.isin(im_ms[:,:,k], -np.inf)
                im_nan = np.isnan(im_ms[:,:,k])
                cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
          
        # calculate cloud cover
        cloud_cover = sum(sum(cloud_mask.astype(int)))/(cloud_mask.shape[0]*cloud_mask.shape[1])
        # the extra image is the 20m SWIR band
        im_extra = im20
    
    return im_ms, georef, cloud_mask, im_extra, imQA

    
def create_jpg(im_ms, cloud_mask, date, satname, filepath):
    """
    Saves a .jpg file with the RGB image as well as the NIR and SWIR1 grayscale images.
    
    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        date: str
            String containing the date at which the image was acquired
        satname: str
            name of the satellite mission (e.g., 'L5')
        
    Returns:
    -----------
        Saves a .jpg image corresponding to the preprocessed satellite image
            
    """

    # rescale image intensity for display purposes
    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_NIR = rescale_image_intensity(im_ms[:,:,3], cloud_mask, 99.9)
    im_SWIR = rescale_image_intensity(im_ms[:,:,4], cloud_mask, 99.9)
    
    # make figure
    fig = plt.figure()
    fig.set_size_inches([18,9])
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(im_RGB)
    ax1.set_title(date + '   ' + satname, fontsize=16)
    
#    if im_RGB.shape[1] > 2*im_RGB.shape[0]:
#        ax1 = fig.add_subplot(311)
#        ax2 = fig.add_subplot(312)
#        ax3 = fig.add_subplot(313)
#    else:
#        ax1 = fig.add_subplot(131)
#        ax2 = fig.add_subplot(132)
#        ax3 = fig.add_subplot(133)        
#    # RGB
#    ax1.axis('off')
#    ax1.imshow(im_RGB)
#    ax1.set_title(date + '   ' + satname, fontsize=16)
#    # NIR
#    ax2.axis('off')
#    ax2.imshow(im_NIR, cmap='seismic')
#    ax2.set_title('Near Infrared', fontsize=16)
#    # SWIR
#    ax3.axis('off')
#    ax3.imshow(im_SWIR, cmap='seismic')
#    ax3.set_title('Short-wave Infrared', fontsize=16)
    
    # save figure
    plt.rcParams['savefig.jpeg_quality'] = 100
    fig.savefig(os.path.join(filepath,
                             date + '_' + satname + '.jpg'), dpi=150) 
    plt.close()
      
    
def save_jpg(metadata, settings):
    """
    Saves a .jpg image for all the images contained in metadata.
    
    KV WRL 2018

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded
        settings: dict
            contains the following fields:
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in the image that is accepted
        'sitename': string
            name of the site (also name of the folder where the images are stored)
                    
    Returns:
    -----------
            
    """
    
    sitename = settings['inputs']['sitename']
    cloud_thresh = settings['cloud_thresh']
    
    # create subfolder to store the jpg files
    filepath_jpg = os.path.join(os.getcwd(), 'data', sitename, 'jpg_files', 'preprocessed')
    try:
        os.makedirs(filepath_jpg)
    except:
        print('')
            
    # loop through satellite list
    for satname in metadata.keys():
        
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
            
        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, imQA = preprocess_single(fn, satname)
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > cloud_thresh or cloud_cover == 1:
                continue
            # save .jpg with date and satellite in the title
            date = filenames[i][:10]
            create_jpg(im_ms, cloud_mask, date, satname, filepath_jpg)
                
def get_reference_sl_manual(metadata, settings):
    """
    Allows the user to manually digitize a reference shoreline that is used seed the shoreline 
    detection algorithm. The reference shoreline helps to detect the outliers, making the shoreline
    detection more robust.
    
    KV WRL 2018

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded
        settings: dict
            contains the following fields:
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in the image that is accepted
        'sitename': string
            name of the site (also name of the folder where the images are stored)
        'output_epsg': int
            epsg code of the desired spatial reference system
                    
    Returns:
    -----------
        reference_shoreline: np.array
            coordinates of the reference shoreline that was manually digitized
            
    """
    
    sitename = settings['inputs']['sitename']
    
    # check if reference shoreline already exists in the corresponding folder
    filepath = os.path.join(os.getcwd(), 'data', sitename)
    filename = sitename + '_reference_shoreline.pkl'
    if filename in os.listdir(filepath):
        print('Reference shoreline already exists and was loaded')
        with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'rb') as f:
            refsl = pickle.load(f)
        return refsl
            
    else:
        # first try to use S2 images (10m res for manually digitizing the reference shoreline)
        if 'S2' in metadata.keys():
            satname = 'S2'
            filepath = SDS_tools.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames']
        # if no S2 images, try L8  (15m res in the RGB with pansharpening)
        elif not 'S2' in metadata.keys() and 'L8' in metadata.keys():
            satname = 'L8'
            filepath = SDS_tools.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames']  
        # if no S2 images and no L8, use L5 images (L7 images have black diagonal bands making it 
        # hard to manually digitize a shoreline)
        elif not 'S2' in metadata.keys() and not 'L8' in metadata.keys() and 'L5' in metadata.keys():
            satname = 'L5'
            filepath = SDS_tools.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames'] 
        else:
            print('You cannot digitize the shoreline on L7 images, add another L8, S2 or L5 to your dataset.') 
        
        # loop trhough the images
        for i in range(len(filenames)):
            
            # read image
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            im_ms, georef, cloud_mask, im_extra, imQA = preprocess_single(fn, satname)
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue
            # rescale image intensity for display purposes
            im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            # plot the image RGB on a figure
            fig = plt.figure()
            fig.set_size_inches([18,9])
            fig.set_tight_layout(True)
            plt.axis('off')
            plt.imshow(im_RGB)
            # decide if the image if good enough for digitizing the shoreline
            plt.title('click <keep> if image is clear enough to digitize the shoreline.\n' +
                      'If not (too cloudy) click on <skip> to get another image', fontsize=14)
            keep_button = plt.text(0, 0.9, 'keep', size=16, ha="left", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))   
            skip_button = plt.text(1, 0.9, 'skip', size=16, ha="right", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            mng = plt.get_current_fig_manager()                                         
            mng.window.showMaximized()
            # let user click on the image once
            pt_input = ginput(n=1, timeout=1000000, show_clicks=True)
            pt_input = np.array(pt_input)
            # if clicks next to <skip>, show another image
            if pt_input[0][0] > im_ms.shape[1]/2:
                plt.close()
                continue
            else:
                # remove keep and skip buttons
                keep_button.set_visible(False)
                skip_button.set_visible(False)
                # update title (instructions)
                plt.title('Click points along the shoreline every ~500 m.\n' +
                          'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                          fontsize=14)     
                plt.draw()
                # let user click on the shoreline
                pts = ginput(n=50000, timeout=1e9, show_clicks=True)
                pts_pix = np.array(pts)
                
                # interpolate between points and show the output to the user
                pts_pix_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                for k in range(len(pts_pix)-1):             
                    if pts_pix[k,0] < pts_pix[k+1,0]:
                        x = pts_pix[[k,k+1],0]
                        y = pts_pix[[k,k+1],1]
                    else:
                        x = pts_pix[[k+1,k],0]
                        y = pts_pix[[k+1,k],1]   
                    xvals = np.linspace(x[0],x[1],50)
                    yinterp = np.interp(xvals,x,y)                    
                    pts_pix_interp = np.append(pts_pix_interp,
                                               np.transpose(np.array([xvals,yinterp])), axis=0)
                pts_pix_interp = np.delete(pts_pix_interp,0,axis=0)
                plt.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r.', markersize=3)
                plt.title('Saving reference shoreline as ' + sitename + '_reference_shoreline.pkl ...')
                plt.draw()
                ginput(n=1, timeout=5, show_clicks=True)
                plt.close()   
                
                # convert image coordinates to world coordinates
                pts_world = SDS_tools.convert_pix2world(pts_pix_interp[:,[1,0]], georef)
                image_epsg = metadata[satname]['epsg'][i]
                pts_coords = SDS_tools.convert_epsg(pts_world, image_epsg, settings['output_epsg'])
                
                # save the reference shoreline
                filepath = os.path.join(os.getcwd(), 'data', sitename)
                with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'wb') as f:
                    pickle.dump(pts_coords, f)
                print('Reference shoreline has been saved')
                break
            
    return pts_coords

def get_reference_sl_Australia(settings):
    """
    Automatically finds a reference shoreline from a high resolution coastline of Australia 
    (Smartline from Geoscience Australia). It finds the points of the national coastline vector
    that are situated inside the area of interest (polygon).
    
    KV WRL 2018

    Arguments:
    -----------
        settings: dict
            contains the following fields:
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in the image that is accepted
        'sitename': string
            name of the site (also name of the folder where the images are stored)
        'output_epsg': int
            epsg code of the desired spatial reference system
                    
    Returns:
    -----------
        ref_sl: np.array
            coordinates of the reference shoreline found in the shapefile
            
    """    
    
    # load high-resolution shoreline of Australia
    filename = os.path.join(os.getcwd(), 'data', 'shoreline_Australia.pkl')
    with open(filename, 'rb') as f:
        sl = pickle.load(f)    
    # spatial reference system of this shoreline
    sl_epsg = 4283          # GDA94 geographic 
    
    # only select the points that sit inside the area of interest (polygon)
    polygon = settings['inputs']['polygon']
    # spatial reference system of the polygon (latitudes and longitudes)
    polygon_epsg = 4326     # WGS84 geographic
    polygon = SDS_tools.convert_epsg(np.array(polygon[0]), polygon_epsg, sl_epsg)[:,:-1]
    
    # use matplotlib function Path
    path = mpltPath.Path(polygon)
    sl_inside = sl[np.where(path.contains_points(sl))]
        
    # convert to desired output coordinate system
    ref_sl = SDS_tools.convert_epsg(sl_inside, sl_epsg, settings['output_epsg'])[:,:-1]
    
    # make a figure for quality control
    plt.figure()
    plt.axis('equal')
    plt.xlabel('Eastings [m]')
    plt.ylabel('Northings [m]')
    plt.plot(ref_sl[:,0], ref_sl[:,1], 'r.')
    polygon = SDS_tools.convert_epsg(polygon, sl_epsg, settings['output_epsg'])[:,:-1]
    plt.plot(polygon[:,0], polygon[:,1], 'k-')
    
    return ref_sl