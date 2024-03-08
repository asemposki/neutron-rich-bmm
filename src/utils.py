# import necessary packages
import numpy as np
import scipy.integrate as scint
from scipy.interpolate import interp1d
import numdifftools as ndt

# define the speed of sound function 
def speed_of_sound(dens, pressure, edens, sat=False, integrate='forward', sampled=False):

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
    sat : bool
        Starting at saturation density (0.16 fm^-3)
        or not. Default is False.
    integrate : str
        Decision to integrate forward or backward.
        Default is 'forward'.
    sampled: bool
        If using samples from the speed of sound, run
        the std and mean using nanmean and nanstd from
        numpy instead of computing envelopes.
        Default is 'False'. 
    
    Returns:
    --------
    cs2 : dict
        The dictonary of results for the 
        speed of sound and the lower and upper
        bounds of it at one sigma.
    '''

    # check for saturation point integration
    if sat is True:
        dens_arr = np.linspace(0.16, 16.0, 600)
    else:
        dens_arr = dens
        
    # using samples
    if sampled is True:
        pres = np.asarray(pressure['samples'])   # (nB, n_samples) shape
        edens_0 = edens['mean']
        
        # huge list for all sampled curves
        edens_full = []
        p_dens_arr = []
        
        # collect the function together
        dn = dens[1] - dens[0]    # equally spaced
        dens_part = dn / dens**2.0   # array of size n
        
        # interpolation and integration for each sample 
        for i in range(len(pres.T)):
            
            # empty list for storing (re-initialize to dump old data)
            en_samples = np.zeros(len(pres))
            
            # outer term (not changing with n)
            outer = (edens_0/dens[0])

            for j in range(len(dens)):
                        
                # try dot product as a simple approximation
                en_samples[j] = dens[j] * (outer + np.dot(pres[:j, i], dens_part[:j]))
                
                # Simpson's Rule integration
                #en_samples[j] = dens[j] * (outer + scint.simps((pres[:j+1, i]/dens[:j+1]**2.0), dens[:j+1]))

            edens_full.append(en_samples)   # shape (n_samples, nB)
                        
        # now calculate chemical potential and derivative
        mu_samples = np.asarray([(np.asarray(edens_full)[i,:] + pres[:,i]) for \
                                 i in range(len(edens_full))])
        log_mu_samples = np.log(mu_samples) # shape (n_samples, nB)
                 
        # speed of sound and return
        cs2_log_samples = dens * np.gradient(log_mu_samples, dn, axis=1, edge_order=2) - np.ones(len(dens))
  
        # get mean, std_dev estimations out, store and return
        cs2_log_mean = np.nanmean(cs2_log_samples, axis=0)
        cs2_log_std = np.nanstd(cs2_log_samples, axis=0)
        
        cs2_log = {
            'mean': cs2_log_mean,
            'std': cs2_log_std,
            'samples': cs2_log_samples
        }
        
        return cs2_log, edens_full
    
    # extract the necessary information
    p_mean = pressure['mean']
    p_low = pressure['mean'] - pressure['std_dev']
    p_high = pressure['mean'] + pressure['std_dev']

    # extract the parameters for edens
    e_mean = edens['mean']
    e_low = edens['lower']
    e_high = edens['upper']
    
    # define constant
    n0 = 0.16    # fm^-3

    # calculate the interpolants
    p_mean_interp = interp1d(dens, (p_mean), kind='cubic', \
                            fill_value='extrapolate')
    p_lower_interp = interp1d(dens, (p_low), kind='cubic', \
                            fill_value='extrapolate')
    p_upper_interp = interp1d(dens, (p_high), kind='cubic', \
                            fill_value='extrapolate')
    
    # define internal functions for integration
    def pres_mean(n):
        return p_mean_interp(n) / (n)**2.0
    def pres_lower(n):
        return p_lower_interp(n) / (n)**2.0
    def pres_upper(n):
        return p_upper_interp(n) / (n)**2.0

    # perform integration
    en_mean = [] #np.zeros_like(p_mean)
    en_lower = []
    en_upper = []
        
    # integrating forwards
    if integrate == 'forward':
        
#        for j in range(len(dens_arr)):
        
        # Simpson's Rule integration
#             en_mean[j] = dens_arr[j] * (e_mean/dens_arr[0] + \
#                                         scint.simps((np.asarray(p_mean)[:j+1]/dens_arr[:j+1]**2.0), dens_arr[:j+1]))
#         print('We did Simpsons integration!')

        for n in dens_arr:
            en_mean.append(n*(e_mean/dens_arr[0] + \
                            scint.quad(lambda x : pres_mean(x), dens_arr[0], n, epsabs=1e-10, epsrel=1e-10)[0]))
            
            en_lower.append(n*(e_low/dens_arr[0] + \
                            scint.quad(lambda x : pres_lower(x), dens_arr[0], n, epsabs=1e-10, epsrel=1e-10)[0]))
            en_upper.append(n*(e_high/dens_arr[0] + \
                            scint.quad(lambda x : pres_upper(x), dens_arr[0], n, epsabs=1e-10, epsrel=1e-10)[0]))
                               
    # try integrating backwards
    elif integrate == 'backward':
        
        for n in dens_arr:
            en_mean.append(n*(e_mean/dens_arr[-1] - \
                            scint.quad(lambda x : pres_mean(x), n, dens_arr[-1], epsabs=1e-10, epsrel=1e-10)[0]))
            en_lower.append(n*(e_low/dens_arr[-1] - \
                            scint.quad(lambda x : pres_lower(x), n, dens_arr[-1], epsabs=1e-10, epsrel=1e-10)[0]))
            en_upper.append(n*(e_high/dens_arr[-1] - \
                            scint.quad(lambda x : pres_upper(x), n, dens_arr[-1], epsabs=1e-10, epsrel=1e-10)[0]))
        
    # dict of energy densities
    edens_int = {
        'mean': en_mean,
        'lower': en_lower,
        'upper': en_upper
    }

    # calculate deriv of pressure
    dpdn_mean = ndt.Derivative(p_mean_interp, step=1e-6, method='central')
    dpdn_lower = ndt.Derivative(p_lower_interp, step=1e-6, method='central')
    dpdn_upper = ndt.Derivative(p_upper_interp, step=1e-6, method='central')
    
    # calculate deriv of energy density
    dedn_mean = np.gradient(edens_int['mean'], dens_arr, edge_order=2)
    dedn_lower = np.gradient(edens_int['lower'], dens_arr, edge_order=2)
    dedn_upper = np.gradient(edens_int['upper'], dens_arr, edge_order=2)
    
    # calculate the chemical potential
    mu_mean = (en_mean + p_mean_interp(dens_arr))/dens_arr
    mu_lower = (en_lower + p_lower_interp(dens_arr))/dens_arr
    mu_upper = (en_upper + p_upper_interp(dens_arr))/dens_arr
    
    # calculate the log of the chemical potential
    log_mu_mean = np.log(mu_mean)
    log_mu_lower = np.log(mu_lower)
    log_mu_upper = np.log(mu_upper)

    # calculate speed of sound using energy density 
    # derivative at desired density array
    cs2_mean = dpdn_mean(dens_arr) / dedn_mean
    cs2_lower = dpdn_lower(dens_arr) / dedn_upper
    cs2_upper = dpdn_upper(dens_arr) / dedn_lower
    
    # calculate speed of sound using chemical potential
    # at desired density array
    cs2_mu_mean = dpdn_mean(dens_arr) / mu_mean
    cs2_mu_lower = dpdn_lower(dens_arr) / mu_upper
    cs2_mu_upper = dpdn_upper(dens_arr) / mu_lower
    
    # calculate speed of sound using log(mu)
    # at desired density array
    cs2_log_mean = dens_arr * np.gradient(log_mu_mean, dens_arr, edge_order=2)
    cs2_log_lower = dens_arr * np.gradient(log_mu_lower, dens_arr, edge_order=2)
    cs2_log_upper = dens_arr * np.gradient(log_mu_upper, dens_arr, edge_order=2)
    
    # collect into dict and return
    cs2 = {
        'mean' : cs2_mean,
        'lower' : cs2_lower,
        'upper' : cs2_upper
    }
    
    # collect other method and return
    cs2_mu = {
        'mean' : cs2_mu_mean,
        'lower' : cs2_mu_lower,
        'upper' : cs2_mu_upper
    }
    
    # collect log method and return
    cs2_log = {
        'mean': cs2_log_mean,
        'lower': cs2_log_lower,
        'upper': cs2_log_upper
    }
    
    # collect mu into dict and return
    mu_dict = {
        'mean':mu_mean,
        'lower':mu_lower,
        'upper':mu_upper
    }

    return dens_arr, cs2, cs2_mu, cs2_log, edens_int, mu_dict