import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline, BSpline, splrep, splev

# helper function to take FRG contour and make into data and make greedy estimates
def frg_greedy_data(plot=False, larger_errors=False):
    
    df_frg = pd.read_csv('../data/frg_data_pressure.csv')
    frg_anm = {
        'dens': df_frg['n/n0'],
        'pressure': 0.94*df_frg['P/P_free']  # scaling the data for ANM
    }

    # cut the contour up to find mean value
    uppercontour = interp1d(frg_anm['dens'][22:], frg_anm['pressure'][22:], kind='cubic', fill_value='extrapolate')
    lowercontour = interp1d(frg_anm['dens'][:22], frg_anm['pressure'][:22], kind='cubic', fill_value='extrapolate')

    # solve at needed points below 
    new_grid = np.linspace(min(frg_anm['dens'][22:]), max(frg_anm['dens'][22:]), 1000)
    uppercontourgrid = uppercontour(new_grid)
    lowercontourgrid = lowercontour(new_grid)

    # finding mean line within the contour (easiest to use for more points, can draw random ones later)
    mean_vals = np.zeros(len(new_grid))
    std_vals = np.zeros(len(new_grid))

    # find last value in array for std_vals here
    if larger_errors is True:
        std_last = (uppercontourgrid[-1]-lowercontourgrid[-1])/2.0
        
        # use it for every value in the set
        std_vals = std_last*np.ones(len(new_grid))
    
    for i in range(len(new_grid)):
        mean_vals[i] = (lowercontourgrid[i] + uppercontourgrid[i])/2.0
        if larger_errors is False:
            std_vals[i] = (uppercontourgrid[i]-lowercontourgrid[i])/2.0

    # discretize this to make it not 100 points ... OR make more points ...
    frg_data = {
        'dens': new_grid,
        'mean': mean_vals,
        'std': std_vals,
        'cov': np.diag(np.square(std_vals))
    }

    if plot is True:
        plt.plot(df_frg['n/n0'], 0.94*df_frg['P/P_free'], color='m', marker='.', linestyle=' ', \
           zorder=10, label='Leonhardt et al. (2020)')
        plt.plot(new_grid, mean_vals, 'g')
        plt.plot(new_grid, uppercontourgrid, 'b')
        plt.plot(new_grid, lowercontourgrid, 'r')
        plt.fill_between(new_grid, mean_vals-std_vals, mean_vals+std_vals, alpha=0.3)
        plt.xscale('log')
        plt.ylim(0.0, 1.2)
        plt.xlim(0.25, 100.0)
        plt.show()

    return frg_data, lowercontour, uppercontour


def log_full_spline(x:np.array, data:np.array, x_select=None, order=3):
   
    # old fashioned way for any other than cubic
    if order != 3:
        t, c, k = splrep(x, data, k=order)

        if x_select is not None:
            return splev(x_select, (t, c, k))
        else:
            return splev(x)
    
    # cubic spline
    else:
        spline = CubicSpline(np.log(x), np.log(data), extrapolate=True)

        if x_select is not None:
            return spline(np.log(x_select))
        else:
            return spline(np.log(x))


def full_cubic_spline(x:np.array, data:np.array, extrapolate=True):
            
    cubic_spline = CubicSpline(data['density'], \
                                            data['mean'], extrapolate=extrapolate)
    
    # results stored and picked from (not going to be very smooth here)
    cubic_results = cubic_spline(x)
    deriv_cubic_results = np.gradient(cubic_spline(x), x, edge_order=2)

    # figure out switching point
    loc = np.where([x[i] <= data['density'][-1] for i in range(len(x))])[0][-1]
    
    # get matching derivative and function information
    m = deriv_cubic_results[loc]
    b = cubic_results[loc]

    # function for the linear case
    def linear(x, m, b):
        return m * (x - x[loc]) + b
        
    # results
    linear_results = linear(x, m, b)

    # make array of these duders
    results = np.zeros(len(x))

    # get the matched linear function at the switching point
    for i in range(len(x)):
        if x[i] <= data['density'][-1]:
            results[i] = cubic_results[i]
        elif x[i] > data['density'][-1]:
            results[i] = linear_results[i]
        
    return results
