"""This module contains all the functions needed for extracting satellite-derived shorelines (SDS)
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.filters as filters 
import skimage.exposure as exposure
import skimage.transform as transform
import sklearn.decomposition as decomposition
import skimage.measure as measure
import skimage.morphology as morphology

# machine learning modules
from sklearn.externals import joblib
from shapely.geometry import LineString

# other modules
from osgeo import gdal, ogr, osr
import scipy.interpolate as interpolate
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib import gridspec
from pylab import ginput
import pickle

# own modules
import SDS_tools, SDS_preprocess

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

 
def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).
    
    KV WRL 2018

    Arguments:
    -----------
        im1, im2: np.array
            Images (2D) with which to calculate the ND index
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        
    Returns:    -----------
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

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates a range of features on the image that are used for the supervised classification.
    The features include spectral normalized-difference indices and standard deviation of the image.
    
    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_bool: np.array
            2D array of boolean indicating where on the image to calculate the features
        
    Returns:    -----------
        features: np.array
            matrix containing each feature (columns) calculated for all 
            the pixels (rows) indicated in im_bool
    """
    
    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # NIR-G
    im_NIRG = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G
    im_SWIRG = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask) 
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask) 
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask) 
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask) 
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  SDS_tools.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    im_std = SDS_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1) 
    
    return features
    
def classify_image_NN(im_ms, im_extra, cloud_mask, min_beach_size, satname):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0
    
    The classifier is a Neural Network, trained on several sites in New South Wales, Australia.
    
    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            Pansharpened RGB + downsampled NIR and SWIR
        im_extra:
            only used for Landsat 7 and 8 where im_extra is the panchromatic band
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        min_beach_size: int
            minimum number of pixels that have to be connected in the SAND class
                
    Returns:    -----------
        im_classif: np.array
            2D image containing labels
        im_labels: np.array of booleans
            3D image containing a boolean image for each class (im_classif == label)

    """     
    
    if satname == 'L5':
        # load classifier (without panchromatic band)
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_nopan.pkl'))
        # calculate features
        n_features = 9
        im_features = np.zeros((im_ms.shape[0], im_ms.shape[1], n_features))
        im_features[:,:,[0,1,2,3,4]] = im_ms
        im_features[:,:,5] = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) # (NIR-G)
        im_features[:,:,6] = nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask) # ND(NIR-R)
        im_features[:,:,7] = nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask) # ND(B-R)
        im_features[:,:,8] = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask) # ND(SWIR-G)
        vec_features = im_features.reshape((im_ms.shape[0] * im_ms.shape[1], n_features))
        
    elif satname in ['L7','L8']:
        # load classifier (with panchromatic band)
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_withpan.pkl'))        
        # calculate features
        n_features = 10
        im_features = np.zeros((im_ms.shape[0], im_ms.shape[1], n_features))
        im_features[:,:,[0,1,2,3,4]] = im_ms
        im_features[:,:,5] = im_extra
        im_features[:,:,6] = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) # (NIR-G)
        im_features[:,:,7] = nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask) # ND(NIR-R)
        im_features[:,:,8] = nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask) # ND(B-R)
        im_features[:,:,9] = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask) # ND(SWIR-G)
        vec_features = im_features.reshape((im_ms.shape[0] * im_ms.shape[1], n_features))
        
    elif satname == 'S2':
        # load classifier (special classifier for Sentinel-2 images)
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_S2.pkl'))
        # calculate features
        vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
        vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0
              
    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]
    
    # classify pixels
    labels = clf.predict(vec_features)
    
    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1])) 
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_size, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_size, connectivity=2)
    
    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1) 
        
    return im_classif, im_labels

def find_wl_contours1(im_ndwi, cloud_mask):
    """
    Traditional method for shorelien detection. 
    Finds the water line by thresholding the Normalized Difference Water Index and applying 
    the Marching Squares Algorithm to contour the iso-value corresponding to the threshold.
    
    KV WRL 2018

    Arguments:
    -----------
        im_ndwi: np.ndarray
            Image (2D) with the NDWI (water index)
        cloud_mask: np.ndarray
            2D cloud mask with True where cloud pixels are
                
    Returns:    -----------
        contours_wl: list of np.arrays 
            contains the (row,column) coordinates of the contour lines

    """  
    
    # reshape image to vector
    vec_ndwi = im_ndwi.reshape(im_ndwi.shape[0] * im_ndwi.shape[1])
    vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec = vec_ndwi[~vec_mask]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    contours = measure.find_contours(im_ndwi, t_otsu)
    
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours = contours_nonans
    
    return contours

def find_wl_contours2(im_ms, im_labels, cloud_mask, buffer_size):
    """
    New robust method for extracting shorelines. Incorporates the classification component to
    refine the treshold and make it specific to the sand/water interface.
    
    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        buffer_size: int
            size of the buffer around the sandy beach over which the pixels are considered in the
            thresholding algorithm.
                
    Returns:    -----------
        contours_wi: list of np.arrays 
            contains the (row,column) coordinates of the contour lines extracted from the 
            NDWI (Normalized Difference Water Index) image
        contours_mwi: list of np.arrays 
            contains the (row,column) coordinates of the contour lines extracted from the 
            MNDWI (Modified Normalized Difference Water Index) image
            
    """  
    
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)
    
    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)
    
    # create a buffer around the sandy beach
    se = morphology.disk(buffer_size)
    im_buffer = morphology.binary_dilation(im_labels[:,:,0], se)
    vec_buffer = im_buffer.reshape(nrows*ncols)
    
    # select water/sand/swash pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
    int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]
    
    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_sand) > 0:
        if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
            int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
        else:
            int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]        
            
    # threshold the sand/water intensities 
    int_all = np.append(int_water,int_sand, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])
    t_wi = filters.threshold_otsu(int_all[:,1])    
    
    # find contour with MS algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi, t_mwi)
    
    # remove contour points that are NaNs (around clouds)
    contours = contours_wi
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours_wi = contours_nonans
    # repeat for MNDWI contours
    contours = contours_mwi
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours_mwi = contours_nonans
        
    return contours_wi, contours_mwi

def process_shoreline(contours, georef, image_epsg, settings):
    """
    Converts the contours from image coordinates to world coordinates. This function also removes
    the contours that are too small to be a shoreline (based on the parameter
    settings['min_length_sl'])
    
    KV WRL 2018

    Arguments:
    -----------
        contours: np.array or list of np.array
            image contours as detected by the function find_contours
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        settings: dict
            contains important parameters for processing the shoreline:
                output_epsg: output spatial reference system
                min_length_sl: minimum length of shoreline perimeter to be kept (in meters)
                reference_sl: [optional] reference shoreline coordinates
                max_dist_ref: max distance (in meters) allowed from a reference shoreline
                
    Returns:    -----------
        shoreline: np.array 
            array of points with the X and Y coordinates of the shoreline
        
    """   
    
    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    # remove contours that have a perimeter < min_length_wl (provided in settings dict)
    # this enable to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))
    
    # if reference shoreline has been manually digitised
    if 'refsl' in settings.keys():
        # only keep the points that are at a certain distance (define in settings) from the 
        # reference shoreline, enables to avoid false detections and remove obvious outliers
        temp = np.zeros((len(contours_array))).astype(bool)
        for k in range(len(settings['refsl'])): 
            temp = np.logical_or(np.linalg.norm(contours_array - settings['refsl'][k,[0,1]],
                                                axis=1) < settings['max_dist_ref'], temp) 
        shoreline = contours_array[temp]
    else:
        shoreline = contours_array
    
    return shoreline

def show_detection(im_ms, cloud_mask, im_labels, shoreline,image_epsg, georef,
                   settings, date, satname):
    """
    Shows the detected shoreline to the user for visual quality control. The user can select "keep"
    if the shoreline detection is correct or "skip" if it is incorrect. 
    
    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        shoreline: np.array 
            array of points with the X and Y coordinates of the shoreline
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        settings: dict
            contains important parameters for processing the shoreline
        date: string
            date at which the image was taken
        satname: string
            indicates the satname (L5,L7,L8 or S2)
                       
    Returns:    -----------
        skip_image: boolean
            True if the user wants to skip the image, False otherwise.
            
    """  
    
    sitename = settings['inputs']['sitename']
    
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection 
    filepath = os.path.join(os.getcwd(), 'data', sitename, 'jpg_files', 'detection')
    
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    
    # compute classified image 
    im_class = np.copy(im_RGB)
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]
        
    # compute MNDWI grayscale image
    im_mwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    
    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:,[0,1]], georef)
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
        
    # according to the image shape, decide whether it is better to have the images in the subplot
    # in different rows or different columns
    fig = plt.figure()
    if im_RGB.shape[1] > 2*im_RGB.shape[0]:
        # vertical subplots
        gs = gridspec.GridSpec(3, 1)
        gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[2,0])
    else: 
        # horizontal subplots
        gs = gridspec.GridSpec(1, 3)
        gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    btn_keep = plt.text(0, 0.9, 'keep', size=16, ha="left", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k',fc='w'))   
    btn_skip = plt.text(1, 0.9, 'skip', size=16, ha="right", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k',fc='w'))
    ax1.set_title(sitename + '    ' + date + '     ' + satname, fontweight='bold', fontsize=16)

    # create image 2 (classification)
    ax2.imshow(im_class)
    ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='--', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1, 0.5), fontsize=9)
    # create image 3 (MNDWI)
    ax3.imshow(im_mwi, cmap='bwr')
    ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    
# additional options
#    ax1.set_anchor('W')
#    ax2.set_anchor('W')
#    cb = plt.colorbar()
#    cb.ax.tick_params(labelsize=10)
#    cb.set_label('MNDWI values')
#    ax3.set_anchor('W')
    
    fig.set_size_inches([12.53, 9.3])
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()
    
    # wait for user's selection: <keep> or <skip>
    pt = ginput(n=1, timeout=100000, show_clicks=True)
    pt = np.array(pt)
    # if user clicks around the <skip> button, return skip_image = True
    if pt[0][0] > im_ms.shape[1]/2:
        skip_image = True
        plt.close()
    else:
        skip_image = False
        btn_skip.set_visible(False)
        btn_keep.set_visible(False)
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)
        plt.close()
    
    return skip_image
    

def extract_shorelines(metadata, settings):
    """
    Extracts shorelines from satellite images.
    
    KV WRL 2018
        
    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded
            
        inputs: dict
            contains the following fields:
                sitename: str
                    String containig the name of the site
                polygon: list
                    polygon containing the lon/lat coordinates to be extracted
                    longitudes in the first column and latitudes in the second column
                dates: list of str
                    list that contains 2 strings with the initial and final dates in format 
                    'yyyy-mm-dd' e.g. ['1987-01-01', '2018-01-01']
                sat_list: list of str
                    list that contains the names of the satellite missions to include 
                    e.g. ['L5', 'L7', 'L8', 'S2']
        
    Returns:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
            
    """
    
    sitename = settings['inputs']['sitename']
    
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(os.getcwd(), 'data', sitename, 'jpg_files', 'detection')
    try:
        os.makedirs(filepath_jpg)
    except:
        print('')
    
    # loop through satellite list
    for satname in metadata.keys():
        
        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        # initialise some variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points 
        output_filename = []   # filename of the images from which the shorelines where derived
        output_cloudcover = [] # cloud cover of the images 
        output_geoaccuracy = []# georeferencing accuracy of the images
        output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)
            
        # loop through the images
        for i in range(len(filenames)):
            
            # get image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # preprocess image (cloud mask + pansharpening/downsampling)
            im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname)
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][i]
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue
            
            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = classify_image_NN(im_ms, im_extra, cloud_mask,
                                    settings['min_beach_size'], satname)
            
            # extract water line contours
            # if there aren't any sandy pixels, use find_wl_contours1 (traditional method), 
            # otherwise use find_wl_contours2 (enhanced method with classification)
            try: # use try/except structure for long runs
                if sum(sum(im_labels[:,:,0])) == 0 :
                    # compute MNDWI (SWIR-Green normalized index) grayscale image
                    im_mndwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    # find water contourson MNDWI grayscale image
                    contours_mwi = find_wl_contours1(im_mndwi, cloud_mask)
                else:
                    # use classification to refine threshold and extract sand/water interface
                    contours_wi, contours_mwi = find_wl_contours2(im_ms, im_labels, 
                                                cloud_mask, settings['buffer_size'])
            except:
                continue
            
            # process water contours into shorelines
            shoreline = process_shoreline(contours_mwi, georef, image_epsg, settings)
            
            if settings['check_detection']:
                date = filenames[i][:10]
                skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                            image_epsg, georef, settings, date, satname)
                if skip_image:
                    continue
            
            # fill and save outputput structure
            output_timestamp.append(metadata[satname]['dates'][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            
        output[satname] = {
                'timestamp': output_timestamp,
                'shoreline': output_shoreline,
                'filename': output_filename,
                'cloudcover': output_cloudcover,
                'geoaccuracy': output_geoaccuracy,
                'idxkeep': output_idxkeep
                }

    # add some metadata
    output['meta'] = {
            'timestamp': 'UTC time',
            'shoreline': 'coordinate system epsg : ' + str(settings['output_epsg']),
            'cloudcover': 'calculated on the cropped image',
            'geoaccuracy': 'RMSE error based on GCPs',
            'idxkeep': 'indices of the images that were kept to extract a shoreline'
            }
    
    # save outputput structure as output.pkl
    filepath = os.path.join(os.getcwd(), 'data', sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)
        
    return output