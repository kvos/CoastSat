"""
This module contains utilities for SDS_Benchmark
    
"""

# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
import pytz
import pdb

# other modules
from scipy import interpolate
from scipy import integrate
from scipy import stats
from scipy import signal
from astropy.timeseries import LombScargle

# Global variables
DAYS_IN_YEAR = 365.2425
SECONDS_IN_DAY = 24*3600
    
###################################################################################################
# UTILITIES
###################################################################################################

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

def calculate_trend(dates,chainage):
    "calculate long-term trend"
    dates_ord = np.array([_.toordinal() for _ in dates])
    dates_ord = (dates_ord - np.min(dates_ord))/DAYS_IN_YEAR   
    trend, intercept, rvalue, pvalue, std_err = stats.linregress(dates_ord, chainage)
    y = dates_ord*trend+intercept
    return trend, y

def detrend(dates,chainage,trend):
    dates_ord = np.array([_.toordinal() for _ in dates])
    dates_ord = (dates_ord - np.min(dates_ord))/DAYS_IN_YEAR 
    return chainage - dates_ord*trend

###################################################################################################
# SEASONAL/MONTHLY AVERAGING
###################################################################################################

def seasonal_average(dates, chainages):
    "compute seasonal averages"
    # define the 4 seasons
    months = ['-%02d'%_ for _ in np.arange(1,13)]
    seasons = np.array([1,4,7,10])
    season_labels = ['DJF', 'MAM', 'JJA', 'SON']
    # put time-series into a pd.dataframe (easier to process)
    df = pd.DataFrame({'dates': dates, 'chainage':chainages})
    df.set_index('dates', inplace=True) 
    # initialise variables for seasonal averages
    dict_seasonal = dict([])
    for k,j in enumerate(seasons):
        dict_seasonal[season_labels[k]] = {'dates':[], 'chainages':[]}
    dates_seasonal = []
    chainage_seasonal = []
    season_ts = []
    for year in np.unique(df.index.year):
        # 4 seasons: DJF, MMA, JJA, SON
        for k,j in enumerate(seasons):
            # middle date
            date_seas = pytz.utc.localize(datetime(year,j,1))
            # if j == 1: date_seas = pytz.utc.localize(datetime(year-1,12,31))
            # for the first season, take the December data from the year before
            if j == 1:
                chain_seas = np.array(df.loc[str(year-1) + months[(j-1)-1]:str(year) + months[(j-1)+1]]['chainage'])
            else:
                chain_seas = np.array(df.loc[str(year) + months[(j-1)-1]:str(year) + months[(j-1)+1]]['chainage'])
            if len(chain_seas) == 0:
                continue
            else:
                dict_seasonal[season_labels[k]]['dates'].append(date_seas)
                dict_seasonal[season_labels[k]]['chainages'].append(np.mean(chain_seas))
                dates_seasonal.append(date_seas)
                chainage_seasonal.append(np.mean(chain_seas))
                season_ts.append(j)
    # convert chainages to np.array (easier to manipulate than a list)
    for seas in dict_seasonal.keys():
         dict_seasonal[seas]['chainages'] = np.array(dict_seasonal[seas]['chainages'])
                
    return dict_seasonal, dates_seasonal, np.array(chainage_seasonal), np.array(season_ts)

def monthly_average(dates, chainages):
    "compute monthly averages"
    # define the 12 months
    months = ['-%02d'%_ for _ in np.arange(1,13)]
    seasons = np.arange(1,13)
    season_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    # put time-series into a pd.dataframe (easier to process)
    df = pd.DataFrame({'dates': dates, 'chainage':chainages})
    df.set_index('dates', inplace=True) 
    # initialise variables for seasonal averages
    dict_seasonal = dict([])
    for k,j in enumerate(seasons):
        dict_seasonal[season_labels[k]] = {'dates':[], 'chainages':[]}
    dates_seasonal = []
    chainage_seasonal = []
    season_ts = []
    for year in np.unique(df.index.year):
        # 4 seasons: DJF, MMA, JJA, SON
        for k,j in enumerate(seasons):
            # middle date
            date_seas = pytz.utc.localize(datetime(year,j,15))
            if date_seas > dates[-1] - timedelta(days=30):
                break
            try:
                chain_seas = np.array(df.loc[str(year) + months[k]]['chainage'])
            except:
                continue
            if len(chain_seas) == 0:
                continue
            else:
                dict_seasonal[season_labels[k]]['dates'].append(date_seas)
                dict_seasonal[season_labels[k]]['chainages'].append(np.mean(chain_seas))
                dates_seasonal.append(date_seas)
                chainage_seasonal.append(np.mean(chain_seas))
                season_ts.append(j)
    # convert chainages to np.array (easier to manipulate than a list)
    for seas in dict_seasonal.keys():
         dict_seasonal[seas]['chainages'] = np.array(dict_seasonal[seas]['chainages'])
                
    return dict_seasonal, dates_seasonal, np.array(chainage_seasonal), np.array(season_ts)

###################################################################################################
# FREQUENCY ANALYSIS
###################################################################################################

def frequency_grid(time,time_step,n0):
    'define frequency grid for Lomb-Scargle transform'
    T = np.max(time) - np.min(time)
    fmin = 1/(T)
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
    E = integrate.simps(ps, x=freqs, even='avg')
    if len(idx_cut) == 0:
        idx_cut = np.ones(freqs.size).astype(bool)
    # integrate only frequencies above cut-off
    Ec = integrate.simps(ps[idx_cut], x=freqs[idx_cut], even='avg')
    return ps, E, Ec, model

def find_peaks_PSD(dates,ts,settings):
    'find the high frequency peak in the tidal time-series'
    # create frequency grid
    t = np.array([_.timestamp() for _ in dates]).astype('float64')
    time_step = settings['n_days']*SECONDS_IN_DAY
    freqs = frequency_grid(t,time_step,settings['n0'])
    # compute power spectrum
    ps_tide,_,_ = power_spectrum(t,ts,freqs,[])
    # find peaks in spectrum
    idx_peaks,_ = signal.find_peaks(ps_tide, height=0)
    y_peaks = _['peak_heights']
    idx_peaks = idx_peaks[np.flipud(np.argsort(y_peaks))]
    # find the strongest peak at the high frequency (defined by freqs_cutoff[1])
    idx_max = idx_peaks[freqs[idx_peaks] > settings['freqs_cutoff']][0]
    # compute the frequencies around the max peak with some buffer (defined by buffer_coeff)
    freqs_max = [freqs[idx_max] - settings['delta_f'], freqs[idx_max] + settings['delta_f']]
    # make a plot of the spectrum
    fig = plt.figure()
    fig.set_size_inches([12,4])
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    ax.grid(linestyle=':', color='0.5')
    ax.plot(freqs,ps_tide)
    ax.set_title('$\Delta t$ = %d days'%settings['n_days'], x=0, ha='left')
    ax.set(xticks=[(DAYS_IN_YEAR*SECONDS_IN_DAY)**-1, (30*SECONDS_IN_DAY)**-1, (16*SECONDS_IN_DAY)**-1, (8*SECONDS_IN_DAY)**-1],
                   xticklabels=['1y','1m','16d','8d']);
    # show top 3 peaks
    for k in range(2):
        ax.plot(freqs[idx_peaks[k]], ps_tide[idx_peaks[k]], 'ro', ms=4)
        ax.text(freqs[idx_peaks[k]], ps_tide[idx_peaks[k]]+1, '%.1f d'%((freqs[idx_peaks[k]]**-1)/(3600*24)),
                ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='square', ec='k',fc='w', alpha=0.5))
    ax.axvline(x=freqs_max[1], ls='--', c='0.5')
    ax.axvline(x=freqs_max[0], ls='--', c='0.5')
    ax.axvline(x=(2*settings['n_days']*SECONDS_IN_DAY)**-1, ls='--', c='k')
    return freqs_max

