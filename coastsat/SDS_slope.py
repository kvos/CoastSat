"""
This module contains all the functions needed to estimate beach slopes

Author: Kilian Vos
"""

# load modules
import os, pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import lines
from scipy import stats as sstats
from scipy import integrate as sintegrate
from scipy import signal as ssignal
from scipy import interpolate as sinterpolate
from astropy.timeseries import LombScargle
import pytz
import pyfes
import pdb
import json
import xarray as xr

###################################################################################################
# Tide functions
###################################################################################################

def compute_tide(coords,date_range,time_step,ocean_tide,load_tide):
    'compute time-series of water level for a location and dates using a time_step'
    # list of datetimes (every timestep)
    dates = []
    date = date_range[0]
    while date <= date_range[1]:
        dates.append(date)
        date = date + timedelta(seconds=time_step)
    # convert list of datetimes to numpy dates
    dates_np = np.empty((len(dates),), dtype='datetime64[us]')
    for i,date in enumerate(dates):
        dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
    lons = coords[0]*np.ones(len(dates))
    lats = coords[1]*np.ones(len(dates))
    # compute heights for ocean tide and loadings
    ocean_short, ocean_long, _ = pyfes.evaluate_tide(ocean_tide,dates_np,lons,lats,num_threads=1)
    load_short, load_long, _ = pyfes.evaluate_tide(load_tide,dates_np,lons,lats,num_threads=1)
    # sum up all components and convert from cm to m
    tide_level = (ocean_short + ocean_long + load_short + load_long)/100
    
    return dates, tide_level

def compute_tide_dates(coords,dates,ocean_tide,load_tide):
    'compute time-series of water level for a location and dates (using a dates vector)'
    dates_np = np.empty((len(dates),), dtype='datetime64[us]')
    for i,date in enumerate(dates):
        dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
    lons = coords[0]*np.ones(len(dates))
    lats = coords[1]*np.ones(len(dates))
    # compute heights for ocean tide and loadings
    ocean_short, ocean_long, _ = pyfes.evaluate_tide(ocean_tide,dates_np,lons,lats,num_threads=1)
    load_short, load_long, _ = pyfes.evaluate_tide(load_tide,dates_np,lons,lats,num_threads=1)
    # sum up all components and convert from cm to m
    tide_level = (ocean_short + ocean_long + load_short + load_long)/100
    
    return tide_level

def compute_tidal_range(centroid, ocean_tide, load_tide):
    'calculate mean tidal range'
    date_range = [pytz.utc.localize(_) for _ in [datetime(2010,1,1), datetime(2010,12,31)]]
    dates_1year, tides_1year = compute_tide(centroid, date_range, 1800, ocean_tide, load_tide)
    # mean high water
    idx_peaks,_ = ssignal.find_peaks(tides_1year, height=0)
    y_peaks = _['peak_heights']
    mean_high_water = np.mean(y_peaks)
    # mean low water
    idx_peaks,_ = ssignal.find_peaks(-tides_1year, height=0)
    y_peaks = _['peak_heights']
    mean_low_water = np.mean(-y_peaks)
    # mean tide range
    tide_range = mean_high_water-mean_low_water
    return np.round(tide_range,3)

def find_tide_data(dates_tide, tide, dates_sub):
    'get the tide data at specific dates in a very efficient way' 
    # nested function
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    # for each date querried find the closest tide point
    tide_sub = []
    for i,date in enumerate(dates_sub):
        if date < dates_tide[0]:
            tide_sub.append(np.nan)
        else:
            print('\rFinding tides: %d%%' % int((i+1)*100/len(dates_sub)), end='')
            tide_sub.append(tide[find(min(item for item in dates_tide if item > date), dates_tide)])
    return np.array(tide_sub)

def plot_tide(dates, tide_level, dates_sub, tide_level_sub):
    'plot water levels'
    q75_wl = np.quantile(tide_level,0.75, interpolation='nearest')
    q25_wl = np.quantile(tide_level,0.25, interpolation='nearest')
    fig = plt.figure()
    fig.set_tight_layout(True)
    fig.set_size_inches([12,4])
    ax = plt.subplot(211)
    ax.set_title('Sub-sampled tide levels')
    q75_wl = np.quantile(tide_level,0.75, interpolation='nearest')
    q25_wl = np.quantile(tide_level,0.25, interpolation='nearest')
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.plot(dates, tide_level, '-', color='0.6')
    ax.plot(dates_sub, tide_level_sub, '-o', color='k', ms=4, mfc='w',lw=1)
    ax.set_ylabel('tide level [m]')
    ax.set_ylim(get_min_max(tide_level))
    ax = plt.subplot(212)
    ax.hist(tide_level, bins=np.arange(np.nanmin(tide_level), np.nanmax(tide_level) + 0.05, 0.05),
             color='0.6', density=True, alpha=1, ec='k', label='all tide')
    ax.hist(tide_level_sub, bins=np.arange(np.nanmin(tide_level_sub), np.nanmax(tide_level_sub) + 0.05, 0.05),
             color='C1', density=True, alpha=0.3, ec='k', label='sub-sampled tide')
    ax.axvline(x=q25_wl, ls='--', color='k', lw=2)
    ax.axvline(x=q75_wl, ls='--', color='k', lw=2)
    ax.legend()
    ax.set_xlabel('tide level [m]')
    ax.set_xlim(get_min_max(tide_level))
    ax.set_ylabel('PDF')

###################################################################################################
# Shoreline time-series functions
###################################################################################################

def range_slopes(min_slope, max_slope, delta_slope):
    'create list of beach slopes to test'
    beach_slopes = []
    slope = min_slope
    while slope < max_slope:
        beach_slopes.append(slope)
        slope = slope + delta_slope
    beach_slopes.append(slope)
    beach_slopes = np.round(beach_slopes,len(str(delta_slope).split('.')[1]))
    return beach_slopes

def make_seasonal(dates, amplitude, freq):
    'make sinusoidal time-series of shoreline change'
    dates_float = np.array([_.timestamp() for _ in dates])
    return amplitude*np.sin(2*np.pi*freq*dates_float)

def tide_correct(chain,tide_level,beach_slopes):
    'apply tidal correction with a range of slopes'
    tsall = []
    for i,slope in enumerate(beach_slopes):
        # apply tidal correction
        tide_correction = (tide_level)/slope
        ts = chain + tide_correction
        tsall.append(ts)
    return tsall

def plot_chain(dates,chain):
    'plot synthetic shoreline time-series'
    fig = plt.figure()
    fig.set_size_inches([12,8])
    fig.set_tight_layout(True)
    counter = 0
    for i,key in enumerate(list(chain.keys())):
        ax = fig.add_subplot(len(chain.keys())+1,1,i+1)
        ax.set_title(key)
        ax.grid(linestyle=':', color='0.5')
        ax.plot(dates, chain[key], 'C0-o', ms=4, mfc='w', lw=1)
        if counter == 0:
            composite = chain[key]
        else:
            composite = composite + chain[key]
        counter = counter + 1
    ax = fig.add_subplot(len(chain.keys())+1,1,i+2)
    ax.grid(linestyle=':', color='0.5')
    ax.plot(dates, composite, 'k-o', ms=4, mfc='w', lw=1)   
    ax.set_title('Composite signal')

def plot_ts(dates,tsall,beach_slopes):
    'plot tidally-corrected timeseries'
    fig = plt.figure()
    fig.set_size_inches([12,4])
    fig.set_tight_layout(True)    
    cmap = cm.get_cmap('RdYlGn')
    color_list = cmap(np.linspace(0,1,len(beach_slopes)))
    ax = fig.add_subplot(111)
    for i in range(len(tsall)):
        ax.plot(dates, tsall[i], '-', color=color_list[i,:])
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.set_ylabel('cross-shore change [m]')
    ax.set_ylim(get_min_max2(tsall))
    ax.set_title('Tidally-corrected time-series')
    # colorbar
    fig, ax = plt.subplots(figsize=(12,1.5))
    fig.set_tight_layout(True) 
    cb = colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')
    cb.ax.set(xlabel='slope values [-]', xticklabels=['0','0.05','0.1','0.15','0.20','0.25'])
    
def plot_cross_distance(dates, cross_distance):
    'plot the time-series of shoreline change from CoastSat'

    for i,key in enumerate(cross_distance.keys()):
        idx_nan = np.isnan(cross_distance[key])
        chain = cross_distance[key][~idx_nan]
        dates_temp = [dates[k] for k in np.where(~idx_nan)[0]]
        if len(chain)==0 or sum(idx_nan) > 0.5*len(idx_nan): continue
        fig,ax=plt.subplots(1,1,figsize=[12,2])
        fig.set_tight_layout(True)
        ax.grid(linestyle=':', color='0.5')
        ax.plot(dates_temp, chain - np.mean(chain), '-o', ms=3, mfc='w', mec='C0')
        ax.set(ylabel='distance [m]')
        ax.text(0.5,0.95,'Transect ' + key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
                va='top', transform=ax.transAxes, fontsize=14)

###################################################################################################
# Power spectrum functions
###################################################################################################
    
def frequency_grid(time,time_step,n0):
    'define frequency grid for Lomb-Scargle transform'
    T = np.max(time) - np.min(time)
    fmin = 1/T
    fmax = 1/(2*time_step) # Niquist criterium
    df = 1/(n0*T)
    N = np.ceil((fmax - fmin)/df).astype(int)
    freqs = fmin + df * np.arange(N)
    return freqs

def power_spectrum(t,y,freqs,idx_cut):
    'compute power spectrum and integrate'
    model = LombScargle(t, y, dy=None, fit_mean=True, center_data=True, nterms=1, normalization='psd')
    ps = model.power(freqs)
    # integrate the entire power spectrum
    E = sintegrate.simpson(ps, x=freqs)
    if len(idx_cut) == 0:
        idx_cut = np.ones(freqs.size).astype(bool)
    # integrate only frequencies above cut-off
    Ec = sintegrate.simpson(ps[idx_cut], x=freqs[idx_cut])
    return ps, E, Ec

def power_spectrum_fft(y,time_step):
    'compute power spectrum and integrate'    
    ps_temp = np.abs(np.fft.fft(y))**2
    freqs_temp = np.fft.fftfreq(y.size, time_step)
    idx_sorted = np.argsort(freqs_temp)
    freqs_sorted = freqs_temp[idx_sorted]
    ps_sorted = ps_temp[idx_sorted] 
    idx_pos = freqs_sorted >= 0
    freqs_fft = freqs_sorted[idx_pos]
    ps_fft = ps_sorted[idx_pos]
    return ps_fft, freqs_fft

def find_tide_peak(dates,tide_level,settings):
    'find the high frequency peak in the tidal time-series'
    # create frequency grid 
    t = np.array([_.timestamp() for _ in dates]).astype('float64')
    days_in_year = 365.2425
    seconds_in_day = 24*3600
    time_step = settings['n_days']*seconds_in_day
    freqs = frequency_grid(t,time_step,settings['n0'])
    # compute power spectrum
    ps_tide,_,_ = power_spectrum(t,tide_level,freqs,[])
    # find peaks in spectrum
    idx_peaks,_ = ssignal.find_peaks(ps_tide, height=0)
    y_peaks = _['peak_heights']
    idx_peaks = idx_peaks[np.flipud(np.argsort(y_peaks))]
    # find the strongest peak at the high frequency (defined by freq_cutoff)
    idx_max = idx_peaks[np.logical_and(freqs[idx_peaks] > settings['freq_cutoff'],
                                       freqs[idx_peaks] < freqs[-1]-settings['delta_f'])][0] 
    # compute the frequencies around the max peak with some buffer (defined by buffer_coeff)
    freqs_max = [freqs[idx_max] - settings['delta_f'], freqs[idx_max] + settings['delta_f']] 
    # only make a plot of the spectrum if plot_bool is True
    if settings['plot_fig']:
        fig = plt.figure()
        fig.set_size_inches([12,4])
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax.grid(linestyle=':', color='0.5')
        ax.plot(freqs,ps_tide)
        ax.set_title('Power Spectrum Density of tide time-series - $\Delta t$ = %d days (Nyquist limit = %d days)'%(settings['n_days'],2*settings['n_days']),
                     x=0, ha='left')
        ax.set(xticks=[(days_in_year*seconds_in_day)**-1, (30*seconds_in_day)**-1, (16*seconds_in_day)**-1, (8*seconds_in_day)**-1],
                       xticklabels=['1y','1m','16d','8d'])
        ax.set(ylabel='PSD',xlabel='frequency')
        # show top 3 peaks
        for k in range(3):
            ax.plot(freqs[idx_peaks[k]], ps_tide[idx_peaks[k]], 'ro', ms=4)
            ax.text(freqs[idx_peaks[k]], ps_tide[idx_peaks[k]]+1, '%.1f d'%((freqs[idx_peaks[k]]**-1)/(3600*24)),
                    ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='square', ec='k',fc='w', alpha=0.5))
        # draw some reference lines
        ax.axvline(x=freqs_max[1], ls='--', c='0.5')
        ax.axvline(x=freqs_max[0], ls='--', c='0.5')
        ax.axvline(x=settings['freq_cutoff'], ls='--', c='k')
        ax.axvline(x=freqs[-1]-settings['delta_f'], ls='--', c='k')
    return freqs_max 

def integrate_power_spectrum(dates_rand,tsall,settings,key=None):
    'integrate power spectrum at the frequency band of peak tidal signal'
    t = np.array([_.timestamp() for _ in dates_rand]).astype('float64')
    seconds_in_day = 24*3600
    time_step = settings['n_days']*seconds_in_day
    freqs = frequency_grid(t,time_step,settings['n0'])    
    beach_slopes = range_slopes(settings['slope_min'], settings['slope_max'], settings['delta_slope'])
    # integrate power spectrum
    idx_interval = np.logical_and(freqs >= settings['freqs_max'][0], freqs <= settings['freqs_max'][1]) 
    E = np.zeros(beach_slopes.size)
    for i in range(len(tsall)):
        ps, _, _ = power_spectrum(t,tsall[i],freqs,[])
        E[i] = sintegrate.simpson(ps[idx_interval], x=freqs[idx_interval])
    # calculate confidence interval
    delta = settings['delta_slope']/2
    f = sinterpolate.interp1d(beach_slopes, E, kind='linear')
    beach_slopes_interp = range_slopes(settings['slope_min'],settings['slope_max'],delta)
    E_interp = f(beach_slopes_interp)
    # find values below minimum + 5%
    slopes_min = beach_slopes_interp[np.where(E_interp <= np.min(E)*(1+settings['prc_conf']))[0]]
    if len(slopes_min) > 1:
        ci = [slopes_min[0],slopes_min[-1]]
    else:
        ci = [beach_slopes[np.argmin(E)],beach_slopes[np.argmin(E)]]
    
    # plot energy vs slope curve
    if settings['plot_fig']:
        # plot energy vs slope curve
        fig = plt.figure()
        fig.set_size_inches([12,4])
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax.grid(linestyle=':', color='0.5')
        ax.set(title='Transect %s - Energy in tidal frequency band for each slope'%key,
               xlabel='Slope values',ylabel='Energy')
        ax.plot(beach_slopes_interp,E_interp,'-k',lw=1.5)
        cmap = cm.get_cmap('RdYlGn')
        color_list = cmap(np.linspace(0,1,len(beach_slopes)))
        for i in range(len(beach_slopes)): ax.plot(beach_slopes[i], E[i],'o',ms=8,mec='k',mfc=color_list[i,:])
        ax.plot(beach_slopes[np.argmin(E)],np.min(E),'bo',ms=14,mfc='None',mew=2)
        ax.text(0.65,0.85,
                'slope estimate = %.3f\nconf. band = [%.4f , %.4f]'%(beach_slopes[np.argmin(E)],ci[0],ci[1]),
                transform=ax.transAxes,va='center',ha='left',
                bbox=dict(boxstyle='round', ec='k',fc='w', alpha=0.5),fontsize=12)
        ax.axhspan(ymin=np.min(E),ymax=np.min(E)*(1+settings['prc_conf']),fc='0.7',alpha=0.5)
        ybottom = ax.get_ylim()[0]
        ax.plot([ci[0],ci[0]],[ybottom,f(ci[0])],'k--',lw=1,zorder=0)
        ax.plot([ci[1],ci[1]],[ybottom,f(ci[1])],'k--',lw=1,zorder=0)
        ax.plot([ci[0],ci[1]],[ybottom,ybottom],'k--',lw=1,zorder=0)
        
    return beach_slopes[np.argmin(E)], ci

def plot_spectrum_all(dates_rand,composite,tsall,settings, slope_est):
    'plot the spectrum of the tidally-corrected time-series of shoreline change'
    t = np.array([_.timestamp() for _ in dates_rand]).astype('float64')
    seconds_in_day = 24*3600
    days_in_year = 365.2425
    time_step = settings['n_days']*seconds_in_day
    freqs = frequency_grid(t,time_step,settings['n0'])    
    beach_slopes = range_slopes(settings['slope_min'], settings['slope_max'], settings['delta_slope'])
    # colormaps
    cmap = cm.get_cmap('RdYlGn')
    color_list = cmap(np.linspace(0,1,len(beach_slopes)))
    indices = np.arange(0,len(beach_slopes))
    
    # make figure 1
    # fig = plt.figure()
    # fig.set_size_inches([12,5])
    # fig.set_tight_layout(True)
    # # axis labels
    # freq_1month = 1/(days_in_year*seconds_in_day/12)
    # xt = freq_1month/np.array([12,3,1, 20/(days_in_year/12), 16/(days_in_year/12)])
    # xl = ['1y', '3m', '1m', '20d', '16d']
    # # loop for plots
    # ax = fig.add_subplot(111)
    # ax.grid(which='major', linestyle=':', color='0.5')
    # ax.set(xticks=xt, xticklabels=xl, title='Power Spectrum of tidally-corrected time-series', ylabel='amplitude')
    # for i,idx in enumerate(indices):
    #     # compute spectrum
    #     ps,_,_ = power_spectrum(t,tsall[idx],freqs,[])
    #     ax.plot(freqs, ps, '-', color=color_list[idx,:], lw=1)
    # # draw some references
    # ax.axvline(x=settings['freqs_max'][0], ls='--', c='0.5')
    # ax.axvline(x=settings['freqs_max'][1], ls='--', c='0.5')
    # ax.axvline(x=freqs[-1]-settings['delta_f'], ls='--', c='k')
    
    # make figure 2
    fig = plt.figure()
    fig.set_size_inches([12,5])
    fig.set_tight_layout(True)
    # axis labels
    xt = 1./(np.flipud(np.arange(settings['n_days']*2,21,1))*24*3600)
    xl = ['%d d'%(_) for _ in np.flipud(np.arange(settings['n_days']*2,21,1))]
    # loop for plots
    ax = fig.add_subplot(111)
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.set(xticks=xt, xticklabels=xl, ylabel='amplitude', title='Inset into the tidal peak frequency bands')
    idx_interval = np.logical_and(freqs >= settings['freqs_max'][0], freqs <= settings['freqs_max'][1])
    for i,idx in enumerate(indices):
        # compute spectrum
        ps, _,_ = power_spectrum(t,tsall[idx],freqs,[])
        ax.plot(freqs[idx_interval], ps[idx_interval], '-', color=color_list[idx,:], lw=1) 
    # non-corrected time-series
    ps,_,_ = power_spectrum(t,composite,freqs,[])
    ax.plot(freqs[idx_interval], ps[idx_interval], '--', color='k', lw=1.5)   
    # true slope
    idx_true = np.where(beach_slopes == slope_est)[0][0]
    ps,_,_ = power_spectrum(t,tsall[idx_true],freqs,[])
    ax.plot(freqs[idx_interval], ps[idx_interval], '--', color='b', lw=1.5)
    # add legend
    nc_line = lines.Line2D([],[],ls='--', c='k', lw=1.5, label='non-corrected time-series')
    true_line = lines.Line2D([],[],ls='--', c='b', lw=1.5, label='min. energy slope (%.3f)'%slope_est)
    ax.legend(handles=[nc_line, true_line], loc=2)
    
###################################################################################################
# Utilities
###################################################################################################

def get_min_max(y):
    'get min and max of a time-series'
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    ymax = np.max([np.abs(ymin),np.abs(ymax)])
    ymin = -np.max([np.abs(ymin),np.abs(ymax)])
    return [ymin,ymax]

def get_min_max2(tsall): 
    'get min and max of a list of tidally-corrected time-series'
    xmin = 1e10
    xmax = -1e10
    for ts in tsall:
        if np.min(ts) < xmin:
            xmin = np.min(ts)
        if np.max(ts) > xmax:
            xmax = np.max(ts)  
    xmax = np.max([np.abs(xmin),np.abs(xmax)])
    xmin = -np.max([np.abs(xmin),np.abs(xmax)]) 
    return [xmin, xmax]

def get_min_max3(cross_distance): 
    'get min and max of a dictionary of tidally-corrected time-series'
    xmin = 1e10
    xmax = -1e10
    for key in cross_distance.keys():
        ts = cross_distance[key] - np.nanmedian(cross_distance[key])
        if np.nanmin(ts) < xmin:
            xmin = np.nanmin(ts)
        if np.nanmax(ts) > xmax:
            xmax = np.nanmax(ts)  
    xmax = np.max([np.abs(xmin),np.abs(xmax)])
    xmin = -np.max([np.abs(xmin),np.abs(xmax)]) 
    return [xmin, xmax]

def plot_timestep(dates, timestep=8):
    'plot the distribution of the timestep for given dates'
    seconds_in_day = 3600*24
    t = np.array([_.timestamp() for _ in dates]).astype('float64')
    delta_t = np.diff(t)
    fig, ax = plt.subplots(1,1,figsize=(12,3))
    fig.set_tight_layout(True)
    ax.grid(which='major', linestyle=':', color='0.5')
    bins = np.arange(np.min(delta_t)/seconds_in_day, np.max(delta_t)/seconds_in_day+1,1)-0.5
    ax.hist(delta_t/seconds_in_day, bins=bins, ec='k', width=1);
    ax.set(xlabel='timestep [days]', ylabel='counts', xticks=timestep*np.arange(0,20),
           xlim=[-1,50], title='Timestep distribution');

def get_region_from_geojson(fn):
    """
    Load the region to be clipped from a geojson file.

    PS 2025

    Arguments:
    -----------
    fn: str
        string containing the path to the file

    Returns:
    -----------
        The coordinates from the geojson

    """
    with open(fn) as f:
        geojson = json.load(f)
    return geojson["features"][0]["geometry"]

def clip_model_to_region(nc_files, geometry, output_dir):
    """
    Clips NetCDF files to the specified region and saves the clipped files.
    Longitudes are kept in the 0 to 360 range.

    PS 2025

    Arguments:
    -----------
    nc_files: list
        list containing all of the files from the directory to be clipped
    geometry: dict
        dictionary containing the geometry of the region to be clipped
    output_dir: str
        string representing the directory where the clipped files will be saved

    """
    coords = np.array(geometry["coordinates"][0])
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360

    for file_path in nc_files:
        print(f"Processing: {file_path}")
        ds = xr.open_dataset(file_path, engine="netcdf4")

        # Ensure longitude is in 0 to 360 format
        if ds.lon.min() < 0:
            ds = ds.assign_coords({"lon": (ds.lon % 360)}).sortby("lon")

        # Select the region
        clipped_ds = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max)
        )

        # Preserve metadata
        clipped_ds.attrs = ds.attrs
        for var in clipped_ds.data_vars:
            clipped_ds[var].attrs = ds[var].attrs

        # Save the clipped file to the same name in the output directory
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        clipped_ds.to_netcdf(output_path)
        print(f"Saved clipped file to: {output_path}")
