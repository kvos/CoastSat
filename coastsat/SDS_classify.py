"""This module contains functions to label satellite images and train the CoastSat classifier

   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import pickle
import warnings
warnings.filterwarnings("ignore")
from matplotlib.widgets import LassoSelector
from matplotlib import path

# image processing modules
from skimage.segmentation import flood
from skimage import morphology
from pylab import ginput
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=2)

# CoastSat functions
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

class SelectFromImage(object):
    # initialize lasso selection class
    def __init__(self, ax, implot, color=[1,1,1]):
        self.canvas = ax.figure.canvas
        self.implot = implot
        self.array = implot.get_array()
        xv, yv = np.meshgrid(np.arange(self.array.shape[1]),np.arange(self.array.shape[0]))
        self.pix = np.vstack( (xv.flatten(), yv.flatten()) ).T
        self.ind = []
        self.im_bool = np.zeros((self.array.shape[0], self.array.shape[1]))
        self.color = color
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, verts):
        # find pixels contained in the lasso
        p = path.Path(verts)
        self.ind = p.contains_points(self.pix, radius=1)
        # color selected pixels
        array_list = []
        for k in range(self.array.shape[2]):
            array2d = self.array[:,:,k]    
            lin = np.arange(array2d.size)
            new_array2d = array2d.flatten()
            new_array2d[lin[self.ind]] = self.color[k]
            array_list.append(new_array2d.reshape(array2d.shape))
        self.array = np.stack(array_list,axis=2)
        self.implot.set_data(self.array)
        self.canvas.draw_idle()
        # update boolean image with selected pixels
        vec_bool = self.im_bool.flatten()
        vec_bool[lin[self.ind]] = 1
        self.im_bool = vec_bool.reshape(self.im_bool.shape)

    def disconnect(self):
        self.lasso.disconnect_events()

def label_images(metadata,settings):
    """
    Interactively label satellite images and save the training data.

    KV WRL 2018

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded
        settings: dict
            contains the following fields:
        cloud_thresh: float
            value between 0 and 1 indicating the maximum cloud fraction in the image that is accepted
        sitename: string
            name of the site (also name of the folder where the images are stored)
        cloud_mask_issue: boolean
            True if there is an issue with the cloud mask and sand pixels are being masked on the images
        labels: dict
            the label name (key) and label number (value) for each class
        flood_fill: boolean
            True to use the flood_fill functionality when labelling sand pixels
        tolerance: float
            tolerance used for flood fill when labelling the sand pixels
        filepath_train: str
            directory in which to save the labelled data

    Returns:
    -----------

    """
    filepath_train = settings['filepath_train']
    # initialize figure
    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                          sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()

    # loop through satellites
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh'] or cloud_cover == 1:
                continue
            # get individual RGB image
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            im_NDVI = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
            im_NDWI = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
            # initialise labels
            im_viz = im_RGB.copy()
            im_labels = np.zeros([im_RGB.shape[0],im_RGB.shape[1]])
            # show RGB image
            ax.axis('off')  
            ax.imshow(im_RGB)
            implot = ax.imshow(im_viz, alpha=0.6)            
            filename = filenames[i][:filenames[i].find('.')][:-4] 
            ax.set_title(filename)
           
            ##############################################################
            # select image to label
            ##############################################################           
            # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
            # this variable needs to be immuatable so we can access it after the keypress event
            key_event = {}
            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key
            # let the user press a key, right arrow to keep the image, left arrow to skip it
            # to break the loop the user can press 'escape'
            while True:
                btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                fig.canvas.draw_idle()                         
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
                    raise StopIteration('User cancelled labelling images')
                else:
                    plt.waitforbuttonpress()
                    
            # if user decided to skip show the next image
            if skip_image:
                ax.clear()
                continue
            # otherwise label this image
            else:
                ##############################################################
                # digitize sandy pixels
                ##############################################################
                ax.set_title('Click on SAND pixels (flood fill activated, tolerance = %.2f)\nwhen finished press <Enter>'%settings['tolerance'])
                # create erase button, if you click there it delets the last selection
                btn_erase = ax.text(im_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))                
                fig.canvas.draw_idle()
                color_sand = settings['colors']['sand']
                sand_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)     
                    # if user clicks on erase, delete the last selection
                    if seed[0] > 0.95*im_ms.shape[1] and seed[1] < 0.05*im_ms.shape[0]:
                        if len(sand_pixels) > 0:
                            im_labels[sand_pixels[-1]] = 0
                            for k in range(im_viz.shape[2]):                              
                                im_viz[sand_pixels[-1],k] = im_RGB[sand_pixels[-1],k]
                            implot.set_data(im_viz)
                            fig.canvas.draw_idle() 
                            del sand_pixels[-1]
                            
                    # otherwise label the selected sand pixels
                    else:
                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(im_NDVI, (seed[1],seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(im_NDWI, (seed[1],seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_sand = np.logical_and(fill_NDVI, fill_NDWI)
                        im_labels[fill_sand] = settings['labels']['sand'] 
                        sand_pixels.append(fill_sand)
                        # show the labelled pixels
                        for k in range(im_viz.shape[2]):                              
                            im_viz[im_labels==settings['labels']['sand'],k] = color_sand[k]
                        implot.set_data(im_viz)
                        fig.canvas.draw_idle() 
                
                ##############################################################
                # digitize white-water pixels
                ##############################################################
                color_ww = settings['colors']['white-water']
                ax.set_title('Click on individual WHITE-WATER pixels (no flood fill)\nwhen finished press <Enter>')
                fig.canvas.draw_idle() 
                ww_pixels = []                        
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)     
                    # if user clicks on erase, delete the last labelled pixels
                    if seed[0] > 0.95*im_ms.shape[1] and seed[1] < 0.05*im_ms.shape[0]:
                        if len(ww_pixels) > 0:
                            im_labels[ww_pixels[-1][1],ww_pixels[-1][0]] = 0
                            for k in range(im_viz.shape[2]):
                                im_viz[ww_pixels[-1][1],ww_pixels[-1][0],k] = im_RGB[ww_pixels[-1][1],ww_pixels[-1][0],k]
                            implot.set_data(im_viz)
                            fig.canvas.draw_idle()
                            del ww_pixels[-1]
                    else:
                        im_labels[seed[1],seed[0]] = settings['labels']['white-water']  
                        for k in range(im_viz.shape[2]):                              
                            im_viz[seed[1],seed[0],k] = color_ww[k]
                        implot.set_data(im_viz)
                        fig.canvas.draw_idle()
                        ww_pixels.append(seed)
                        
                im_sand_ww = im_viz.copy()
                btn_erase.set(text='<Esc> to Erase', fontsize=12)
                
                ##############################################################
                # digitize water pixels (with lassos)
                ##############################################################
                color_water = settings['colors']['water']
                ax.set_title('Click and hold to draw lassos and select WATER pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle() 
                selector_water = SelectFromImage(ax, implot, color_water)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()                         
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_water.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_water.array = im_sand_ww
                        implot.set_data(selector_water.array)
                        fig.canvas.draw_idle()                         
                        selector_water.implot = implot
                        selector_water.im_bool = np.zeros((selector_water.array.shape[0], selector_water.array.shape[1])) 
                        selector_water.ind=[]          
                # update im_viz and im_labels
                im_viz = selector_water.array
                selector_water.im_bool = selector_water.im_bool.astype(bool)
                im_labels[selector_water.im_bool] = settings['labels']['water']
                
                im_sand_ww_water = im_viz.copy()
                
                ##############################################################
                # digitize land pixels (with lassos)
                ##############################################################
                color_land = settings['colors']['other land features']
                ax.set_title('Click and hold to draw lassos and select OTHER LAND pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle() 
                selector_land = SelectFromImage(ax, implot, color_land)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()                         
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = im_sand_ww_water
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()                         
                        selector_land.implot = implot
                        selector_land.im_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1])) 
                        selector_land.ind=[]
                # update im_viz and im_labels
                im_viz = selector_land.array
                selector_land.im_bool = selector_land.im_bool.astype(bool)
                im_labels[selector_land.im_bool] = settings['labels']['other land features']  
                
                # save labelled image
                ax.set_title(filename)
                fig.canvas.draw_idle()                         
                fp = os.path.join(filepath_train,settings['inputs']['sitename'])
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fig.savefig(os.path.join(fp,filename+'.jpg'), dpi=150)
                ax.clear()
                # save labels and features
                features = dict([])
                for key in settings['labels'].keys():
                    im_bool = im_labels == settings['labels'][key]
                    features[key] = SDS_shoreline.calculate_features(im_ms, cloud_mask, im_bool)
                training_data = {'labels':im_labels, 'features':features, 'label_ids':settings['labels']}
                with open(os.path.join(fp, filename + '.pkl'), 'wb') as f:
                    pickle.dump(training_data,f)
                    
    # close figure when finished
    plt.close(fig)

def load_labels(train_sites, settings):
    
    filepath_train = settings['filepath_train']
    # initialize the features dict
    features = dict([])
    n_features = 20
    first_row = np.nan*np.ones((1,n_features))
    for key in settings['labels'].keys():
        features[key] = first_row
    # loop through each site 
    for site in train_sites:
        sitename = site[:site.find('.')] 
        filepath = os.path.join(filepath_train,sitename)
        if os.path.exists(filepath):
            list_files = os.listdir(filepath)
        else:
            continue
        # make a new list with only the .pkl files (no .jpg)
        list_files_pkl = []
        for file in list_files:
            if '.pkl' in file:
                list_files_pkl.append(file)
        # load and append the training data to the features dict
        for file in list_files_pkl:
            # read file
            with open(os.path.join(filepath, file), 'rb') as f:
                labelled_data = pickle.load(f) 
            for key in labelled_data['features'].keys():
                if len(labelled_data['features'][key])>0: # check that is not empty
                    # append rows
                    features[key] = np.append(features[key],
                                labelled_data['features'][key], axis=0)  
    # remove the first row (initialized with nans)
    print('Number of pixels per class in training data:')
    for key in features.keys(): 
        features[key] = features[key][1:,:]
        print('%s : %d pixels'%(key,len(features[key])))
    
    return features

def format_training_data(features, classes, labels):
    # initialize X and y
    X = np.nan*np.ones((1,features[classes[0]].shape[1]))
    y = np.nan*np.ones((1,1))
    # append row of features to X and corresponding label to y 
    for i,key in enumerate(classes):
        y = np.append(y, labels[i]*np.ones((features[key].shape[0],1)), axis=0)
        X = np.append(X, features[key], axis=0)
    # remove first row
    X = X[1:,:]; y = y[1:]
    # replace nans with something close to 0
    # training algotihms cannot handle nans
    X[np.isnan(X)] = 1e-9 
    
    return X, y

def plot_confusion_matrix(y_true,y_pred,classes,normalize=False,cmap=plt.cm.Blues):
    """
    Function copied from the scikit-learn examples (https://scikit-learn.org/stable/)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]), ylim=[3.5,-0.5],
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    fig.tight_layout()
    return ax

def evaluate_classifier(classifier, metadata, settings):
    """
    Interactively visualise the new classifier.

    KV WRL 2018

    Arguments:
    -----------
        classifier: joblib object
            Multilayer Perceptron to be used for image classification
        metadata: dict
            contains all the information about the satellite images that were downloaded
        settings: dict
            contains the following fields:
        cloud_thresh: float
            value between 0 and 1 indicating the maximum cloud fraction in the image that is accepted
        sitename: string
            name of the site (also name of the folder where the images are stored)
        cloud_mask_issue: boolean
            True if there is an issue with the cloud mask and sand pixels are being masked on the images
        labels: dict
            the label name (key) and label number (value) for each class
        filepath_train: str
            directory in which to save the labelled data

    Returns:
    -----------

    """  
    # create folder
    fp = os.path.join(os.getcwd(), 'evaluation')
    if not os.path.exists(fp):
        os.makedirs(fp)
        
    # initialize figure
    fig,ax = plt.subplots(1,2,figsize=[17,10],sharex=True, sharey=True,constrained_layout=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()  

    # create colormap for labels
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    # loop through satellites
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        
        # load classifiers and
        if satname in ['L5','L7','L8']:
            pixel_size = 15
        elif satname == 'S2':
            pixel_size = 10
        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels
        buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        
        # loop through images
        for i in range(len(filenames)):   
            # image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            image_epsg = metadata[satname]['epsg'][i]
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue
            # calculate a buffer around the reference shoreline (if any has been digitised)
            im_ref_buffer = SDS_shoreline.create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                                    pixel_size, settings)
            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = SDS_shoreline.classify_image_NN(im_ms, im_extra, cloud_mask,
                                    min_beach_area_pixels, classifier)
            # there are two options to map the contours:
            # if there are pixels in the 'sand' class --> use find_wl_contours2 (enhanced)
            # otherwise use find_wl_contours2 (traditional)
            try: # use try/except structure for long runs
                if sum(sum(im_labels[:,:,0])) < 10 :
                    # compute MNDWI image (SWIR-G)
                    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    # find water contours on MNDWI grayscale image
                    contours_mwi = SDS_shoreline.find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
                else:
                    # use classification to refine threshold and extract the sand/water interface
                    contours_wi, contours_mwi = SDS_shoreline.find_wl_contours2(im_ms, im_labels,
                                                cloud_mask, buffer_size_pixels, im_ref_buffer)
            except:
                print('Could not map shoreline for this image: ' + filenames[i])
                continue
            # process the water contours into a shoreline
            shoreline = SDS_shoreline.process_shoreline(contours_mwi, cloud_mask, georef, image_epsg, settings)
            try:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg)[:,[0,1]], georef)
            except:
                # if try fails, just add nan into the shoreline vector so the next parts can still run
                sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
            # make a plot
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            # create classified image
            im_class = np.copy(im_RGB)
            for k in range(0,im_labels.shape[2]):
                im_class[im_labels[:,:,k],0] = colours[k,0]
                im_class[im_labels[:,:,k],1] = colours[k,1]
                im_class[im_labels[:,:,k],2] = colours[k,2]        
            # show images
            ax[0].imshow(im_RGB)
            ax[1].imshow(im_RGB)
            ax[1].imshow(im_class, alpha=0.5)
            ax[0].axis('off')
            ax[1].axis('off')
            filename = filenames[i][:filenames[i].find('.')][:-4] 
            ax[0].set_title(filename)  
            ax[0].plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
            ax[1].plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
            # save figure
            fig.savefig(os.path.join(fp,settings['inputs']['sitename'] + filename[:19] +'.jpg'), dpi=150)
            # clear axes
            for cax in fig.axes:
               cax.clear()
   
    # close the figure at the end
    plt.close()