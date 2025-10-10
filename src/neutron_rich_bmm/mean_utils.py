import numpy as np
from scipy.interpolate import CubicSpline

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
