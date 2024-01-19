# import necessary packages
import numpy as np
import scipy.integrate as scint
from scipy.interpolate import interp1d

# define the speed of sound function 
def speed_of_sound(dens, pressure, edens, scaled=False):

    '''
    Function to evaluate the speed of sound of
    a system given the pressure, number density,
    and initial parameters for the energy
    density integration. 

    Parameters:
    -----------
    dens : numpy 1d array
        The number density of the system.
    pressure : dict
        The dictonary of pressure means
        and standard deviations from the system.
    edens : dict
        The dictionary of energy density 
        means and standard deviations for a 
        specific starting point in density.
    scaled : bool
        Whether the data is scaled or not. If
        True, then the unscaling needs to be
        performed outside the code. Default is False.

    Returns:
    --------
    cs2 : dict
        The dictonary of results for the 
        speed of sound and the lower and upper
        bounds of it at one sigma.
    '''

    # check for scaling (can change later if needed)
    if scaled is True:
        raise ValueError('Cannot unscale data here, \
                        please do this outside of \
                        this function.')

    # extract the necessary information
    p_mean = pressure['mean']
    p_low = pressure['mean'] - pressure['std_dev']
    p_high = pressure['mean'] + pressure['std_dev']

    # extract the parameters for edens
    e_mean = edens['mean']
    e_low = edens['lower']
    e_high = edens['upper']

    # calculate the interpolants
    p_mean_interp = interp1d(dens, (p_mean), kind='cubic', \
                            fill_value='extrapolate')
    p_lower_interp = interp1d(dens, (p_low), kind='cubic', \
                            fill_value='extrapolate')
    p_upper_interp = interp1d(dens, (p_high), kind='cubic', \
                            fill_value='extrapolate')
    
    # define internal functions for integration
    def pres_mean(n):
        return p_mean_interp(n) / n**2.0
    def pres_lower(n):
        return p_lower_interp(n) / n**2.0
    def pres_upper(n):
        return p_upper_interp(n) / n**2.0

    # perform integration
    en_mean = []
    en_lower = []
    en_upper = []

    for n in dens:
        en_mean.append(n*(e_mean/dens[0] + \
                        scint.quad(lambda x : pres_mean(x), dens[0], n)[0]))
        en_lower.append(n*(e_low/dens[0] + \
                        scint.quad(lambda x : pres_lower(x), dens[0], n)[0]))
        en_upper.append(n*(e_high/dens[0] + \
                        scint.quad(lambda x : pres_upper(x), dens[0], n)[0]))
        
    # dict of energy densities
    edens_int = {
        'mean': en_mean,
        'lower': en_lower,
        'upper': en_upper
    }

    # calculate deriv of pressure
    dpdn_mean = np.gradient(p_mean, dens)
    dpdn_lower = np.gradient(p_low, dens)
    dpdn_upper = np.gradient(p_high, dens)
    
    # calculate deriv of energy density
    dedn_mean = np.gradient(edens_int['mean'], dens)
    dedn_lower = np.gradient(edens_int['lower'], dens)
    dedn_upper = np.gradient(edens_int['upper'], dens)

    # calculate speed of sound (think more about uncertainties)
    cs2_mean = dpdn_mean / dedn_mean
    cs2_lower = dpdn_upper / dedn_lower
    cs2_upper = dpdn_lower / dedn_upper

    # collect into dict and return
    cs2 = {
        'mean' : cs2_mean,
        'lower' : cs2_lower,
        'upper' : cs2_upper
    }

    return cs2, edens_int