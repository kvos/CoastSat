"""This module contains functions to analyze the shoreline data along transects' 
    
   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# other modules
import skimage.transform as transform
from pylab import ginput
import geopandas as gpd

# own modules
from coastsat import SDS_tools

def create_transect(origin, orientation, length):
    """
    Create a 2D transect of points with 1m interval. 
    
    Arguments:
    -----------
        origin: np.array
            contains the X and Y coordinates of the origin of the transect
        orientation: int
            angle of the transect (anti-clockwise from North) in degrees
        length: int
            length of the transect in metres
        
    Returns:    
    -----------
        transect: np.array
            contains the X and Y coordinates of the transect
        
    """     
    x0 = origin[0]
    y0 = origin[1]
    # orientation of the transect
    phi = (90 - orientation)*np.pi/180 
    # create a vector with points at 1 m intervals
    x = np.linspace(0,length,length+1)
    y = np.zeros(len(x))
    coords = np.zeros((len(x),2))
    coords[:,0] = x
    coords[:,1] = y 
    # translate and rotate the vector using the origin and orientation
    tf = transform.EuclideanTransform(rotation=phi, translation=(x0,y0))
    transect = tf(coords)
                
    return transect

def draw_transects(output, settings):
    """
    Allows the user to draw shore-normal transects over the mapped shorelines.
    
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        settings: dict
            contains the inputs
    Returns:    
    -----------
        transects: dict
            contains the X and Y coordinates of all the transects drawn. These are also saved
             as a .geojson (+ a .jpg figure showing the location of the transects)
        
    """    
    sitename = settings['inputs']['sitename']
    filepath = os.path.join(settings['inputs']['filepath'], sitename)

    # plot all shorelines
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.axis('equal')
    ax1.set_xlabel('Eastings [m]')
    ax1.set_ylabel('Northings [m]')
    ax1.grid(linestyle=':', color='0.5')
    for i in range(len(output['shorelines'])):
        sl = output['shorelines'][i]
        date = output['dates'][i]
        ax1.plot(sl[:, 0], sl[:, 1], '.', markersize=3, label=date.strftime('%d-%m-%Y'))
#    ax1.legend()
    fig1.set_tight_layout(True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()
    ax1.set_title('Click two points to define each transect (first point is the origin of the transect).\n'+
              'When all transects have been defined, click on <ENTER>', fontsize=16)
    
    # initialise variable
    transects = dict([])
    counter = 0
    # loop until user breaks it by click <enter>
    while 1:
        # let user click two points
        pts = ginput(n=2, timeout=1e9)
        if len(pts) > 0:
            origin = pts[0]
        else:
            fig1.gca().set_title('Transect locations', fontsize=16)
            fig1.savefig(os.path.join(filepath, 'jpg_files', sitename + '_transect_locations.jpg'), dpi=200)
            plt.title('Transect coordinates saved as ' + sitename + '_transects.geojson')
            plt.draw()
            ginput(n=1, timeout=3, show_clicks=True)
            plt.close(fig1)
            break
        counter = counter + 1
        transect = np.array([pts[0], pts[1]])
        
        # alternative of making the transect the origin, orientation and length
#        temp = np.array(pts[1]) - np.array(origin)
#        phi = np.arctan2(temp[1], temp[0])
#        orientation = -(phi*180/np.pi - 90)
#        length = np.linalg.norm(temp)
#        transect = create_transect(origin, orientation, length)
        
        transects[str(counter)] = transect
        
        # plot the transects on the figure
        ax1.plot(transect[:,0], transect[:,1], 'b-', lw=2.5)
        ax1.plot(transect[0,0], transect[0,1], 'rx', markersize=10)
        ax1.text(transect[-1,0], transect[-1,1], str(counter), size=16,
                 bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        
    # save as transects.geojson (for GIS)
    gdf = SDS_tools.transects_to_gdf(transects)
    # set projection
    gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
    # save as geojson    
    gdf.to_file(os.path.join(filepath, sitename + '_transects.geojson'), driver='GeoJSON', encoding='utf-8')
    print('Transect locations saved in ' + filepath)
        
    return transects

def compute_intersection(output, transects, settings):
    """
    Computes the intersection between the 2D mapped shorelines and the transects, to generate
    time-series of cross-shore distance along each transect.
    
    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        transects: dict
            contains the X and Y coordinates of the transects (first and last point needed for each
            transect).
        settings: dict
            contains parameters defining :
                along_dist: alongshore distance to caluclate the intersection (median of points 
                within this distance).      
        
    Returns:    
    -----------
        cross_dist: dict
            time-series of cross-shore distance along each of the transects. These are not tidally 
            corrected.
        
    """      
    shorelines = output['shorelines']
    along_dist = settings['along_dist']
    
    # initialise variables
    chainage_mtx = np.zeros((len(shorelines),len(transects),6))
    idx_points = []
    
    for i in range(len(shorelines)):

        sl = shorelines[i]
        idx_points_all = []
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0,:])
            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
            diff_angle = (phi - phi_sl)
            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]
            idx_points_all.append(idx_close)        
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                chainage_mtx[i,j,:] = np.tile(np.nan,(1,6))
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                    
                # compute mean, median, max, min and std of chainage position
                n_points = len(xy_rot[0,:])
                mean_cross = np.nanmean(xy_rot[0,:])
                median_cross = np.nanmedian(xy_rot[0,:])
                max_cross = np.nanmax(xy_rot[0,:])
                min_cross = np.nanmin(xy_rot[0,:])
                std_cross = np.nanstd(xy_rot[0,:])
                # store all statistics
                chainage_mtx[i,j,:] = np.array([mean_cross, median_cross, max_cross,
                            min_cross, n_points, std_cross])
    
        # store the indices of the shoreline points that were used
        idx_points.append(idx_points_all)
     
    # format into dictionnary
    chainage = dict([])
    chainage['mean'] = chainage_mtx[:,:,0]
    chainage['median'] = chainage_mtx[:,:,1]
    chainage['max'] = chainage_mtx[:,:,2]
    chainage['min'] = chainage_mtx[:,:,3]
    chainage['npoints'] = chainage_mtx[:,:,4]
    chainage['std'] = chainage_mtx[:,:,5]
    chainage['idx_points'] = idx_points
        
    # only return the median
    cross_dist = dict([])
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = chainage['median'][:,j]    
    
    return cross_dist