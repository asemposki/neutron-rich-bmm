import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

# helper function to take FRG contour and make into data
def frg_greedy_data(plot=False):
    
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
    
    for i in range(len(new_grid)):
        mean_vals[i] = (lowercontourgrid[i] + uppercontourgrid[i])/2.0
        std_vals[i] = (uppercontourgrid[i]-lowercontourgrid[i])/2.0

    # discretize this to make it not 100 points
    frg_data = {
        'dens': new_grid[10::28],
        'mean': mean_vals[10::28],
        'std': std_vals[10::28],
        'cov': np.diag(np.square(std_vals[10::28]))
    }

    if plot is True:
        plt.plot(df_frg['n/n0'], 0.94*df_frg['P/P_free'], color='m', marker='.', linestyle=' ', \
           zorder=10, label='Leonhardt et al. (2020)')
        plt.plot(new_grid, mean_vals, 'g')
        plt.plot(new_grid, uppercontourgrid, 'b')
        plt.plot(new_grid, lowercontourgrid, 'r')
        plt.xscale('log')
        plt.ylim(0.0, 1.2)
        plt.xlim(0.25, 100.0)
        plt.show()

    return frg_data


def log_full_cubic_spline(x:np.array, data:np.array, x_select=None):
    
    # new version to accomodate the FRG mock data
    cubic_spline = CubicSpline(np.log(x), np.log(data), extrapolate=True)
    
    if x_select is not None:
        return cubic_spline(np.log(x_select))
    else:
        return cubic_spline(np.log(x))


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
