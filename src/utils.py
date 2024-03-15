# import necessary packages
import numpy as np
import scipy.integrate as scint
from scipy.interpolate import interp1d
import numdifftools as ndt

from pqcd_reworked import PQCD

# define the speed of sound function 
def speed_of_sound(dens, pressure, edens=None, sat=False, integrate='forward', sampled=False):

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
            outer = (edens_0/dens[-1])

            # running integration backward from pQCD
            for j in range(len(dens)):
                        
                # try dot product as a simple approximation
                #en_samples[j] = dens[j] * (outer + np.dot(pres[:j, i], dens_part[:j]))
                
                # Simpson's Rule integration
                en_samples[j] = dens[j] * (outer - scint.simps((pres[j:, i]/dens[j:]**2.0), dens[j:]))

            edens_full.append(en_samples)   # shape (n_samples, nB)
                        
        # now calculate chemical potential and derivative
        mu_samples = np.asarray([((np.asarray(edens_full)[i,:] + pres[:,i]))/dens for \
                                 i in range(len(edens_full))])   # samples, nB

        # get the results using 1/mu dP/dn instead (more stable)
        print(pres.shape)  # nB, samples
        dpdn_samples = np.gradient(pres, dn, axis=0, edge_order=2)
        
        print(dpdn_samples.shape)
        
        cs2_samples = np.asarray([(mu_samples[i,:])**(-1.0) * dpdn_samples[:,i] \
                                  for i in range(len(edens_full))])
        
        # get mean, std_dev estimations out, store and return
        cs2_mean = np.nanmean(cs2_samples, axis=0)
        cs2_std = np.nanstd(cs2_samples, axis=0)
        
        cs2 = {
            'mean': cs2_mean,
            'std': cs2_std,
            'samples': cs2_samples
        }
        
        return cs2, edens_full
    
    # extract the necessary information
    p_mean = pressure['mean']
    p_low = pressure['mean'] - pressure['std_dev']
    p_high = pressure['mean'] + pressure['std_dev']
    
    # extract the parameters for edens (for pqcd these will be full curves)
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
    en_mean = []
    en_lower = []
    en_upper = []
        
    # integrating forwards
    if integrate == 'forward':
        
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
    
    # calculate the chemical potential
    mu_mean = (en_mean + p_mean_interp(dens_arr))/dens_arr
    mu_lower = (en_lower + p_lower_interp(dens_arr))/dens_arr
    mu_upper = (en_upper + p_upper_interp(dens_arr))/dens_arr
    
    # calculate the log of the chemical potential
    log_mu_mean = np.log(mu_mean)
    log_mu_lower = np.log(mu_lower)
    log_mu_upper = np.log(mu_upper)
    
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

    return dens_arr, cs2, cs2_log, edens_int, mu_dict


def boundary_conditions(dens, pres_dict, index=0):
    
    # call pQCD class
    pqcd = PQCD(X=1, Nf=2) # classic implementation here
    
    # constants
    hbarc = 197.327 # Mev fm
    
    # unpack dictionary
    pres_FG = pres_dict['FG']
    pres_NLO = pres_dict['NLO']
    pres_N2LO = pres_dict['N2LO']
    
    # set up new dictionary for eps(n) BCs
    edens_FG = dict()
    edens_NLO = dict()
    edens_N2LO = dict()
    
    # make mu_FG array from the selected density array (no playing around)
    n_q = dens*3.0  # n_q [fm^-3]

    # convert to GeV^3 for mu_q
    conversion_fm3 = ((1000.0)**(3.0))/((197.33)**(3.0)) # [fm^-3]  (do the opposite of this)
    n_q = n_q/conversion_fm3  # [GeV^3]

    # invert to get mu
    _, _, mu_FG = pqcd.inversion(n_mu=n_q)  # [GeV] # these are quark chemical potentials
    mU_FG = mu_FG[:, None]
    
    # FG BCs
    edens_FG['mean'] = ((3.0 / (2 * np.pi**2.0)) * (3.0 * np.pi**2.0 * dens/2.0)**(4.0/3.0) * hbarc)[index]
    edens_FG['lower'] = (dens*3*1000.*mu_FG - (pres_dict_FG['mean']-pres_dict_FG['std_dev']))[index]
    edens_FG['upper'] = (dens*3*1000.*mu_FG - (pres_dict_FG['mean']+pres_dict_FG['std_dev']))[index]
    
    # NLO BCs
    edens_NLO['mean'] = ((pqcd.mu_1(mU_FG)[:,0]*1000.) * 3.0 * dens - \
                         (pres_dict_NLO['mean'] - pres_dict_FG['mean']))[index]
    edens_NLO['lower'] = ((pqcd.mu_1(mU_FG)[:,0]*1000.) * 3.0 * dens \
    - ((pres_dict_NLO['mean'] - pres_dict_FG['mean']) - \
       (pres_dict_NLO['std_dev']-pres_dict_FG['std_dev'])))[index]
    edens_NLO['upper'] = ((pqcd.mu_1(mU_FG)[:,0]*1000.) * 3.0 * dens \
    - ((pres_dict_NLO['mean'] - pres_dict_FG['mean']) + \
       (pres_dict_NLO['std_dev']-pres_dict_FG['std_dev'])))[index]
    
    # N2LO BCs
    edens_N2LO['mean'] = ((pqcd.mu_2(mU_FG)[:,0]*1000.) * 3.0 * dens - \
                          (pres_dict_N2LO['mean'] - pres_dict_NLO['mean']))[index]
    
    edens_N2LO['lower'] = ((pqcd.mu_2(mU_FG)[:,0]*1000.) * 3.0 * dens - \
                           ((pres_dict_N2LO['mean'] - pres_dict_NLO['mean']) - \
                            (pres_dict_N2LO['std_dev']-pres_dict_NLO['std_dev'])))[index]
   
    edens_N2LO['upper'] = ((pqcd.mu_2(mU_FG)[:,0]*1000.) * 3.0 * dens - \
                           ((pres_dict_N2LO['mean'] - pres_dict_NLO['mean']) + \
                            (pres_dict_N2LO['std_dev']-pres_dict_NLO['std_dev'])))[index]
        
    # add corrections to single out each order
    edens_NLO['mean'] += edens_FG['mean']
    edens_NLO['lower'] += edens_FG['lower']
    edens_NLO['upper'] += edens_FG['upper']
    
    edens_N2LO['mean'] += edens_NLO['mean']
    edens_N2LO['lower'] += edens_NLO['lower']
    edens_N2LO['upper'] += edens_NLO['upper']

    # combine into dictionary and return
    edens_dict = {
        'FG': edens_FG,
        'NLO': edens_NLO,
        'N2LO': edens_N2LO
    }
    
    return mu_FG, mU_FG, edens_dict


def pal_eos(kf, cc):
    
    '''
    Python version of PAL EOS for maintaining causality. 
    Coupling constants found via the FORTRAN code paleoscc.f90,
    not included in this function. This function is designed
    to be used as a mean function in the GP for chiral EFT.
    '''
    
    # extract coupling constants from cc dict
    K0 = 260. #cc['K0']
    A = -47.83618 #cc['A']
    B = 31.01158 #cc['B']
    Bp = 0. #cc['Bp']
    Sig = 1.500259  #cc['Sig']
    
    # other constants
    hc = 197.33
    mn = 939.
    kf0 = (1.5*np.pi**2.*0.16)**(1./3.)
    ef0 = (hc*kf0)**2./2./939.
    sufac = (2.**(2./3.)-1.)*0.6*ef0
    s0 = 30.
    
    # other coupling constants
    C1 = -83.841
    C2 = 22.999
    Lambda1 = 1.5*kf0
    Lambda2 = 3.*kf0
    
    # conversion from kf to n to solve that problem
    n = 2.0 * kf**3.0 / (3.0 * np.pi**2.0)
    
    # write it as E/A first and then move to pressure for output
    one = mn * n0 * (n/n0) + (3.0/5.0)*ef0*n0*(n/n0)**(5.0/3.0)
    two = 0.5*A*n0*(n/n0)**2.0 + (B*n0*(n/n0)**(Sig+1.0))/(1.0 + Bp * (n/n0)**(Sig - 1.0))
    sum_1 = C1 * (Lambda1/kf0)**3.0 * ((Lambda1/kf) - np.arctan(Lambda1/kf))
    sum_2 = C2 * (Lambda2/kf0)**3.0 * ((Lambda2/kf) - np.arctan(Lambda2/kf))
    three = 3.0 * n0 * (n/n0) * (sum_1 + sum_2)
    
    eps_kf = one + two + three
    
    # convert to E/A from eps
    enperpart_kf = eps_kf / n
    
    # now calculate pressure using differentiation
    derivEA = np.gradient(enperpart_kf, kf)
    pressure_kf = (n * kf / 3.0) * derivEA
    
    return pressure_kf