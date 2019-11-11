#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# Kilian Vos WRL 2018

#%% 1. Initial settings

# load modules
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
from datetime import datetime, timedelta
import pytz
from scipy import interpolate, stats
from matplotlib import gridspec
from pylab import ginput
from skimage.segmentation import flood, flood_fill
import matplotlib.cm as cm

# plotting params
plt.style.use('default')
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

#%% 2. Sites that will be used for training

# choose the sites (only add new site to improve training data)
filepath_sites = os.path.join(os.getcwd(), 'training_sites')
train_sites = os.listdir(filepath_sites)
print('Sites for training:\n%s'%train_sites)

#filepath = os.path.join(os.getcwd(), 'data')
## dowload images at the sites
#dates = ['2015-01-01', '2019-01-01']
#sat_list = 'L8'
#for site in train_sites[1:]:
#    polygon = SDS_tools.polygon_from_kml(os.path.join(filepath_sites,site))
#    sitename = site[:site.find('.')]  
#    inputs = {'polygon':polygon, 'dates':dates, 'sat_list':sat_list,
#              'sitename':sitename, 'filepath':filepath}
#    print(site)
#    metadata = SDS_download.retrieve_images(inputs)

#%% 3. Create the training data
filepath_train = os.path.join(os.getcwd(), 'training_data')
filepath_images = os.path.join(os.getcwd(), 'data')
settings ={'cloud_thresh':0.1,'cloud_mask_issue':False, 'inputs':{'filepath':filepath_images},
           'labels':{'sand':1,'white-water':2,'water':3,'other land features':4}}
for site in train_sites:
    settings['inputs']['sitename'] = site[:site.find('.')] 
    # load metadata
    metadata = SDS_download.get_metadata(settings['inputs'])
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
            date = filenames[i][:19]
            nrow = im_ms.shape[0]
            ncol = im_ms.shape[1]
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            im_NIR = im_ms[:,:,3]
            im_SWIR = im_ms[:,:,4]
            # initialise labels
            im_viz = im_RGB.copy()
            im_labels = np.zeros([im_RGB.shape[0],im_RGB.shape[1]])
            # show image
            fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                                  sharey=True)
            mng = plt.get_current_fig_manager()                                         
            mng.window.showMaximized()
            ax.axis('off')  
            ax.imshow(im_RGB)
            # add buttons
            keep_button = plt.text(0, 0.9, 'keep', size=16, ha="left", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            skip_button = plt.text(1, 0.9, 'skip', size=16, ha="right", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            # let user click on the image once (keep or skip)
            pt_input = ginput(n=1, timeout=1e9, show_clicks=False)
            pt_input = np.array(pt_input)
            # if clicks next to <skip>, show another image
            if pt_input[0][0] > im_ms.shape[1]/2:
                plt.close()
                continue
            else:
                # remove keep and skip buttons
                keep_button.set_visible(False)
                skip_button.set_visible(False)
                # digitize sandy pixels (using flood_fill)
                ax.set_title('Digitize SAND pixels', fontweight='bold', fontsize=15)
                plt.draw()
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    if len(seed) == 0:
                        break
                    else:
                        seed = tuple(np.round(seed).astype(int)[0])
                        filled = flood(im_SWIR, (seed[1],seed[0]), tolerance=0.05) 
                        im_labels[filled] = settings['labels']['sand']
                        im_viz[im_labels==1,0] = 1
                        im_viz[im_labels==1,1] = 0.65
                        im_viz[im_labels==1,2] = 0
                        ax.imshow(im_viz, alpha=1)
                        plt.draw()  
                        
                # digitize white-water pixels (individually)
                ax.set_title('Digitize WHITE-WATER pixels', fontweight='bold', fontsize=15)
                plt.draw()
                pt_ww = ginput(n=-1, timeout=0, show_clicks=True)
                if len(pt_ww) > 0:
                    pt_ww = np.round(pt_ww).astype(int)
                    for k in range(len(pt_ww)):
                        im_labels[pt_ww[k,1],pt_ww[k,0]] = settings['labels']['white-water']
                    im_viz[im_labels==settings['labels']['white-water'],0] = 1
                    im_viz[im_labels==settings['labels']['white-water'],1] = 0
                    im_viz[im_labels==settings['labels']['white-water'],2] = 1
                    ax.imshow(im_viz, alpha=1)
                    plt.draw()
                         
                # digitize water pixels (with a rectangle)
                ax.set_title('Digitize WATER pixels', fontweight='bold', fontsize=15)
                plt.draw()
                vtc_water = ginput(n=2, timeout=0, show_clicks=True)
                if len(vtc_water) > 0:
                    pt = np.round(vtc_water).astype(int) 
                    idx_row = np.arange(np.min(pt[:,1]),np.max(pt[:,1])+1,1) 
                    idx_col = np.arange(np.min(pt[:,0]),np.max(pt[:,0])+1,1) 
                    xx, yy = np.meshgrid(idx_row,idx_col, indexing='ij')
                    rows = xx.reshape(xx.shape[0]*xx.shape[1])
                    cols = yy.reshape(yy.shape[0]*yy.shape[1])
                    for k in range(len(rows)):
                        im_labels[rows[k],cols[k]] = settings['labels']['water']
                    im_viz[im_labels==settings['labels']['water'],0] = 0
                    im_viz[im_labels==settings['labels']['water'],1] = 0
                    im_viz[im_labels==settings['labels']['water'],2] = 1
                    ax.imshow(im_viz, alpha=0.3)
                    plt.draw() 
                
                # digitize land pixels (with a rectangle)
                ax.set_title('Digitize OTHER LAND pixels', fontweight='bold', fontsize=15)
                plt.draw()
                vtc_land = ginput(n=2, timeout=0, show_clicks=True)
                if len(vtc_land) > 0:
                    pt = np.round(vtc_land).astype(int) 
                    idx_row = np.arange(np.min(pt[:,1]),np.max(pt[:,1])+1,1) 
                    idx_col = np.arange(np.min(pt[:,0]),np.max(pt[:,0])+1,1) 
                    xx, yy = np.meshgrid(idx_row,idx_col, indexing='ij')
                    rows = xx.reshape(xx.shape[0]*xx.shape[1])
                    cols = yy.reshape(yy.shape[0]*yy.shape[1])
                    for k in range(len(rows)):
                        im_labels[rows[k],cols[k]] = settings['labels']['other land features']
                    im_viz[im_labels==settings['labels']['other land features'],0] = 1
                    im_viz[im_labels==settings['labels']['other land features'],1] = 1
                    im_viz[im_labels==settings['labels']['other land features'],2] = 0
                    ax.imshow(im_viz, alpha=0.4)
                    plt.draw()  
                    
                # save image
                filename = filenames[i][:filenames[i].find('.')][:-4] 
                ax.set_title(filename, fontweight='bold', fontsize=15)
                plt.draw()
                fp = os.path.join(filepath_train,settings['inputs']['sitename'])
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fig.savefig(os.path.join(fp,filename+'.jpg'), dpi=200)
                plt.close(fig)
                # save labels and features
                features = dict([])
                for key in settings['labels'].keys():
                    im_bool = im_labels == settings['labels'][key]
                    features[key] = SDS_shoreline.calculate_features(im_ms, cloud_mask, im_bool)
                training_data = {'labels':im_labels, 'features':features, 'label_ids':settings['labels']}
                with open(os.path.join(fp, filename + '.pkl'), 'wb') as f:
                    pickle.dump(training_data,f)
                    
#%% format training data
# initialise a matrix with 20 columns ( for each feature)
n_features = 20
first_row = np.nan*np.ones((1,n_features))
features_matrix = {'sand':first_row, 'white-water':first_row,
                   'water':first_row, 'other land features':first_row}
# read the files
list_files = os.listdir(fp)
list_files_pkl = []
for file in list_files:
    if '.pkl' in file:
        list_files_pkl.append(file)
# load and append the training data to the initialised matrix
for file in list_files_pkl:
    with open(os.path.join(fp, file), 'rb') as f:
        training_data = pickle.load(f)  
        for key in training_data['features'].keys():
            # check if empty
            if len(training_data['features'][key])>0:
                features_matrix[key] = np.append(features_matrix[key],
                            training_data['features'][key], axis=0)  
# remove the first row (initialized with nans)
for key in settings['labels'].keys(): 
    features_matrix[key] = features_matrix[key][1:,:]

with open(os.path.join(os.getcwd(),'training_data', 'CoastSat_training_set_L8.pkl'), 'rb') as f:
    features_matrix = pickle.load(f) 
    
#%% train classifier
# re-sample the land and water classes
n_samples = 7000 # values depends on how many samples they are in the sand class
features_matrix['water'] =  features_matrix['water'][np.random.choice(features_matrix['water'].shape[0],
             n_samples, replace=False),:]
features_matrix['other land features'] =  features_matrix['other land features'][np.random.choice(features_matrix['other land features'].shape[0],
             n_samples, replace=False),:]
    
# combine into X matrix with features and y vector with labels
X = first_row
y = np.nan*np.ones((1,1))
label_names = ['sand','white-water','water','other land features']
labels = [1,2,3,0]
for i,key in enumerate(label_names):
    y = np.append(y, labels[i]*np.ones((features_matrix[key].shape[0],1)), axis=0)
    X = np.append(X, features_matrix[key], axis=0)
X = X[1:,:]
X[np.isnan(X)] = 1e-9
y = y[1:]

# train Neural Network
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer 
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

# divide in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
clf = MLPClassifier(solver='adam')
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

# cross-validation
scores = cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
             
# final classifier
clf_final = MLPClassifier(solver='adam')
clf_final.fit(X,y)
print(clf_final.score(X,y))
# save classifier
#joblib.dump(clf_final, os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_L8_test.pkl'))

#%% test on images
filepath_sites = os.path.join(os.getcwd(), 'training_sites')
train_sites = os.listdir(filepath_sites)
filepath_images = os.path.join(os.getcwd(), 'data')
settings ={'cloud_thresh':0.1,'cloud_mask_issue':False, 'inputs':{'filepath':filepath_images}}
cmap = cm.get_cmap('tab20c')
colorpalette = cmap(np.arange(0,13,1))
colours = np.zeros((3,4))
colours[0,:] = colorpalette[5]
colours[1,:] = np.array([204/255,1,1,1])
colours[2,:] = np.array([0,91/255,1,1])
for site in train_sites:
    settings['inputs']['sitename'] = site[:site.find('.')] 
    # load metadata
    metadata = SDS_download.get_metadata(settings['inputs'])
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
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            # show image
            fig,ax = plt.subplots(1,2,figsize=[17,10], tight_layout=True,sharex=True,
                                  sharey=True)
            mng = plt.get_current_fig_manager()                                         
            mng.window.showMaximized()
            ax[0].axis('off')  
            ax[0].imshow(im_RGB)
            # classify image
            features = SDS_shoreline.calculate_features(im_ms,
                                    cloud_mask, np.ones(cloud_mask.shape).astype(bool))
            features[np.isnan(features)] = 1e-9
            # remove NaNs and clouds
            vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
            vec_nan = np.any(np.isnan(features), axis=1)
            vec_mask = np.logical_or(vec_cloud, vec_nan)
            features = features[~vec_mask, :]
            # predict with NN classifier
            labels = clf_final.predict(features)
            # recompose image
            vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1])) 
            vec_classif[~vec_mask] = labels
            im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))
            # labels
            im_sand = im_classif == 1
            # remove small patches of sand
    #        im_sand = morphology.remove_small_objects(im_sand, min_size=25, connectivity=2)
            im_swash = im_classif == 2
            im_water = im_classif == 3
            im_land = im_classif == 0
            im_labels = np.stack((im_sand,im_swash,im_water), axis=-1) 
            
            # display classified image
            im_class = np.copy(im_RGB)
            for k in range(0,im_labels.shape[2]):
                im_class[im_labels[:,:,k],0] = colours[k,0]
                im_class[im_labels[:,:,k],1] = colours[k,1]
                im_class[im_labels[:,:,k],2] = colours[k,2]
                
            ax[1].imshow(im_class)
            # add buttons
            keep_button = plt.text(0, 0.9, 'next image', size=16, ha="left", va="top",
                                   transform=ax[1].transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            skip_button = plt.text(1, 0.9, 'next site', size=16, ha="right", va="top",
                                   transform=ax[1].transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            # let user click on the image once (keep or skip)
            pt_input = ginput(n=1, timeout=0, show_clicks=False)
            pt_input = np.array(pt_input)
            # if clicks next to <skip>, show another image
            if pt_input[0][0] > im_ms.shape[1]/2:
                break
            else:        
                continue
        