# import necessary packages
import numpy as np
import scipy.integrate as scint
import scipy as sp
from scipy.interpolate import interp1d
import numdifftools as ndt
import matplotlib.pyplot as plt

from pqcd_reworked import PQCD

# global constants
n0 = 0.164
    

# function for obtaining training data for GP implementation
def gp_data(data_xeft, data_pqcd, cutoff=40, all_orders=True, matter='SNM'):
    
    '''
    Helper function for determining training data 
    from the Chiral EFT and pQCD full training sets.
    Used for the BMM when a GP is the method of choice.
    
    Parameters:
    -----------
    data_xeft : dict
        The dictionary of densities, means, and variances
        of the Chiral EFT data.
    
    data_pqcd : dict
        The dictionary of densities, means, and variances 
        of the pQCD data.
    
    cutoff : int
        The scaled density cutoff we are using for
        pQCD data.
        
    all_orders : bool
        Toggle if data is more than one-dimensional.
        Default is True.
    
    Returns:
    --------
    training_set : dict
        The dictionary of selected training data 
        concatenated from both EOSs.
    '''
    
    # split into training and testing data
    n_xeft = data_xeft["density"]
    n_pqcd = data_pqcd["density"]
    
    # toggle
    if all_orders is True:
        p_mean_xeft = data_xeft["mean"][:, -1]
        p_stdv_xeft = data_xeft["std_dev"][:, -1]
        p_cov_xeft = data_xeft["cov"][..., -1]
    else:
        p_mean_xeft = data_xeft["mean"]
        p_stdv_xeft = data_xeft["std_dev"]
        p_cov_xeft = data_xeft["cov"]

    p_mean_pqcd = data_pqcd["mean"][:, -1]
    p_stdv_pqcd = data_pqcd["std_dev"][:, -1]
    p_cov_pqcd = data_pqcd["cov"][..., -1]
    
    # cut into training and testing sets
    # training data
    n_train_xeft = n_xeft[1::2]
    n_train_pqcd = n_pqcd[1::2]

    chiral_train = {
        'dens': n_train_xeft,
        'mean': p_mean_xeft[1::2],
        'std': p_stdv_xeft[1::2],
        'cov': p_cov_xeft[1::2, 1::2]
    }
    pqcd_train = {
        'dens': n_train_pqcd,
        'mean': p_mean_pqcd[1::2],
        'std': p_stdv_pqcd[1::2],
        'cov': p_cov_pqcd[1::2, 1::2]
    }

    # testing data
    n_test_xeft = n_xeft[::2]
    n_test_pqcd = n_pqcd[::2]

    chiral_test = {
        'dens': n_test_xeft,
        'mean': p_mean_xeft[::2],
        'std': p_stdv_xeft[::2],
        'cov': p_cov_xeft[::2,::2]
    }
    pqcd_test = {
        'dens': n_test_pqcd,
        'mean': p_mean_pqcd[::2],
        'std': p_stdv_pqcd[::2],
        'cov': p_cov_pqcd[::2,::2]
    }
    
    # store cutoff in terms of density
    pqcd_dens_cutoff = cutoff * 0.164
    
    # cut the training sets up
    if chiral_train['dens'][-1] > 0.34:
        chiral_cutoff = np.where([chiral_train['dens']>=0.34])[1][0]
    else:
        chiral_cutoff = -1
        
    chiral_tr = {}
    for key,i in chiral_train.items():
        if chiral_train[key].ndim == 1:
            chiral_tr[key] = chiral_train[key][:chiral_cutoff]
        elif chiral_train[key].ndim == 2:
            chiral_tr[key] = chiral_train[key][:chiral_cutoff, :chiral_cutoff]

    # chiral point selection
  #  log_space_chiral = get_linear_mask_in_log_space(chiral_train['dens'], chiral_train['dens'][40],\
   #                                                 chiral_train['dens'][chiral_cutoff], 0.25, base=np.e)
    if matter == 'SNM':
        chiral_tr_final = {}
        for key,i in chiral_tr.items():
            if chiral_tr[key].ndim == 1:
                chiral_tr_final[key] = chiral_tr[key][15::20] #[40::30] # [15::20] works well for full chiral curve
            elif chiral_tr[key].ndim == 2:
                chiral_tr_final[key] = chiral_tr[key][15::20,15::20] #[40::30, 40::30]
#     else:
#         chiral_tr_final = {}
#         for key,i in chiral_tr.items():
#             if chiral_tr[key].ndim == 1:
#                 chiral_tr_final[key] = chiral_tr[key][10::70] 
#             elif chiral_tr[key].ndim == 2:
#                 chiral_tr_final[key] = chiral_tr[key][10::70, 10::70]

    print(chiral_tr_final['dens'].shape, chiral_tr_final['mean'].shape, \
          chiral_tr_final['std'].shape, chiral_tr_final['cov'].shape)

    # pqcd point selection
    if pqcd_train['dens'][-1] < pqcd_dens_cutoff:
        pqcd_cutoff = min(pqcd_train['dens'])
    else:
        pqcd_cutoff = np.where([pqcd_train['dens']>=pqcd_dens_cutoff])[1][0]

    pqcd_tr = {}
    for key,i in pqcd_train.items():
        if pqcd_train[key].ndim == 1:
            pqcd_tr[key] = pqcd_train[key][pqcd_cutoff:]
        elif pqcd_train[key].ndim == 2:
            pqcd_tr[key] = pqcd_train[key][pqcd_cutoff:, pqcd_cutoff:]
    
    # concatenate everything 
    log_space_pqcd = get_linear_mask_in_log_space(pqcd_tr['dens'], pqcd_tr['dens'][0],\
                                                    pqcd_tr['dens'][-1], 0.20, base=np.e)
    pqcd_tr_final = {}
    for key,i in pqcd_tr.items():
        if pqcd_tr[key].ndim == 1:
            pqcd_tr_final[key] = pqcd_tr[key][::50]  # 50 before
        elif pqcd_tr[key].ndim == 2:
            pqcd_tr_final[key] = pqcd_tr[key][::50, ::50]#[log_space_pqcd][:, log_space_pqcd]

    print(chiral_tr_final['dens'].shape, chiral_tr_final['mean'].shape, \
          chiral_tr_final['std'].shape, chiral_tr_final['cov'].shape)
    print(pqcd_tr_final['dens'].shape, pqcd_tr_final['mean'].shape, \
          pqcd_tr_final['std'].shape, pqcd_tr_final['cov'].shape)

    # now concatenate into block diagonal matrix
    training_set = {
        'dens' : np.concatenate((chiral_tr_final['dens'], pqcd_tr_final['dens'])),
        'mean' : np.concatenate((chiral_tr_final['mean'], pqcd_tr_final['mean'])),
        'std' : np.concatenate((chiral_tr_final['std'], pqcd_tr_final['std'])),
        'cov' : sp.linalg.block_diag(chiral_tr_final['cov'], pqcd_tr_final['cov'])
    }

    # print the covariance to check
    print('Cov shape:', training_set['cov'].shape)
    
    return chiral_tr_final, pqcd_tr_final, training_set
    

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
        speed of sound (calculated using 1\mu dP/dn)
        and the lower and upper bounds of it at 
        one sigma.
        
    edens_full : dict
        The energy density dictionary of means and
        variances returned when sampled == True.
        
    dens_arr : numpy.ndarray
        The densities corresponding to the 
        speed of sound calculation (if sat is True, 
        this will reflect from saturation up), returned
        when sampled is False.
        
    cs2_log : dict
        The dict of speed of sound values from using
        the n * dlog(mu)/dn method. Returned when 
        sampled is False.
        
    edens_int : dict
        The dict of energy densities, returned when 
        sampled is False.
        
    mu_dict : dict
        The dict of chemical potential values, returned
        when sampled is False.
    '''

    # check for saturation point integration
    if sat is True:
        dens_arr = np.linspace(0.164, 16.4, 1200)
    else:
        dens_arr = dens
        
    # using samples
    if sampled is True:
        pres = np.asarray(pressure['samples'])   # (nB, n_samples) shape
        edens_0 = edens['samples'] #edens['mean']   ### how did we not catch this we neeeeeed to fix this...
        
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
            outer = (edens_0[i]/dens[-1])  # adding change of integration constant w/each sample

            # running integration backward from pQCD
            for j in range(len(dens)):
                
                # Simpson's Rule integration
                en_samples[j] = dens[j] * (outer - scint.simps((pres[j:, i]/dens[j:]**2.0), dens[j:]))

            edens_full.append(en_samples)   # shape (n_samples, nB)
                        
        # now calculate chemical potential and derivative
        mu_samples = np.asarray([((np.asarray(edens_full)[i,:] + pres[:,i]))/dens for \
                                 i in range(len(edens_full))])   # samples, nB

        # get the results using 1/mu dP/dn instead (more stable)
        #print(pres.shape)  # nB, samples
        dpdn_samples = np.gradient(pres, dn, axis=0, edge_order=2)
        
        #print(dpdn_samples.shape)
        
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
    n0 = 0.164    # fm^-3

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


def select_draws(pressure_dict:dict):

    # pull samples
    samples = pressure_dict['samples']   # [dens, draws]
    mean = pressure_dict['mean']
    std = pressure_dict['std_dev']
    lower_cutoff = mean - std
    upper_cutoff = mean + std

    # craft list for our new samples
    reduced_samples = samples.copy()
    rows_2_remove = []

    # run through all samples and densities
    for i, sample in enumerate(samples.T):
        result = sample < lower_cutoff
        result2 = sample > upper_cutoff
        if True in result or True in result2:
            rows_2_remove.append(i)
    
    valid_samples = np.delete(reduced_samples, rows_2_remove, axis=1)
    
    return np.asarray(valid_samples)

# define function for the cs2 calculation in full
def cs2_routine(gp_dict, sat_cut=None, plot=True, save_data=False):
    
    # unpack the dict
    gp_dens = gp_dict['dens']
    gp_mean = gp_dict['mean']
    gp_std = gp_dict['std_dev']
    
    # set up the samples
    if gp_dict['samples'] is not None and gp_dict['true'] is None:
        pres_samples = gp_dict['samples']
    elif gp_dict['true'] is not None and gp_dict['samples'] is None:
        pres_samples = gp_dict['true']
    else:
        raise ValueError('Too many variables to assign.')
    
    #conversion for speed of sound
    convert_pqcd = np.load('../data/eos_data/pqcd_fg_data_NSM.npz')

    # interpolate for a functional form to use 
    convert_interp = sp.interpolate.interp1d(convert_pqcd['density'], convert_pqcd['mean'], \
                                     kind='cubic', fill_value='extrapolate')

    # run interpolation for FG scaling (needed for determination of P(n) from scaled result)
    gp_cs2_convert_arr = convert_interp(gp_dens)
    
    # convert samples over
    pres_samples_unscaled = [pres_samples[:,i]*gp_cs2_convert_arr \
                             for i in range(len(np.asarray(pres_samples).T))]
    
    pres_dict = {
        'dens': gp_dens[sat_cut:],
        'mean': gp_mean[sat_cut:]*gp_cs2_convert_arr[sat_cut:],
        'std_dev': gp_std[sat_cut:]*gp_cs2_convert_arr[sat_cut:],
        'samples': np.asarray(pres_samples_unscaled)[:, sat_cut:].T
    }
    
    # energy density bounds (not changing)
    en_0 = 43656.4556069574
    en_0_lower = 43510.21143367375
    en_0_upper = 43797.873283570414

    # for draws, try new anchor point function
    pqcd_class = PQCD(X=1, Nf=3)
    edens_0_draw_arr = pqcd_class.anchor_point_edens(pres_dict['samples'], \
                                                     anchor=gp_dens[-1])

    # make dict of values to send to speed of sound code
    edens_dict = {
        'mean': en_0, 
        'lower': en_0_lower,
        'upper': en_0_upper,
        'samples': edens_0_draw_arr
    }
    
    # call speed of sound function (sampled = True automatically runs the integration downwards)
    cs2_sampled, edens_full = speed_of_sound(gp_dens[sat_cut:], pres_dict, \
                                         edens_dict, sat=False, sampled=True)
    
    # get mean and std_dev for edens to plot P(eps)
    edens_mean = np.nanmean(edens_full, axis=0)
    edens_std = np.nanstd(edens_full, axis=0)
    
    # dict entries 
    cs2_sampled_mean = cs2_sampled['mean']
    cs2_sampled_std = cs2_sampled['std']
    cs2_sampled_samples = cs2_sampled['samples']
    
    # plot if desired
    if plot is True:
        cs2_plots(edens_full, pres_dict, cs2_sampled, sat_cut)
        
    # save for plotting later (uncomment to save)
    if save_data is True:
        np.savez('../data/eos_data/cs2_gp_40.npz', dens=gp_dens[sat_cut:], \
                 mean=cs2_sampled['mean'], std=cs2_sampled['std'], samples=cs2_sampled['samples'])
        
    # print statement to let user know it has finished
    print('Woo it is over!')

    return pres_dict, edens_full, cs2_sampled

# plotter for speed of sound and draws
def cs2_plots(edens_full, pres_dict, cs2_sampled, sat_cut):

    # energy density plot
    sat_cut = 0
    density_test = pres_dict['dens']
    [plt.plot(density_test[sat_cut:]/n0, edens_full[i]) for i in range(len(edens_full))]
    plt.xlabel(r'$n/n_{0}$')
    plt.ylabel(r'$\varepsilon(n)$')
    plt.xlim(0.16/n0, 16.4/n0)
    plt.xscale('log')
    plt.show()

    # plot the pressure vs. energy density
    plt.plot(np.nanmean(edens_full, axis=0), pres_dict['mean'])
    plt.fill_between(np.nanmean(edens_full, axis=0), pres_dict['mean']-pres_dict['std_dev'], \
                     pres_dict['mean']+pres_dict['std_dev'], alpha=0.2)
    plt.xlabel(r'$\varepsilon(n)$')
    plt.ylabel(r'$P(n)$')
    plt.show()

    # samples plot
    [plt.plot(density_test[sat_cut:]/n0, cs2_sampled['samples'][i]) for i in range(len(edens_full))]
    plt.axhline(y=1.0/3.0, linestyle='dashed')
    plt.xlabel(r'$n/n_{0}$')
    plt.ylabel(r'$c_{s}^{2}(n)$')
    plt.xlim(0.16/n0, 16.4/n0)
    plt.xscale('log')
    plt.ylim(0.0, 0.75)
    plt.show()

    # mean and std dev band plot
    [plt.plot(density_test[sat_cut:]/n0, cs2_sampled['samples'][i], alpha=0.1) \
     for i in range(len(edens_full))]
    plt.plot(density_test[sat_cut:]/n0, cs2_sampled['mean'], 'k', zorder=20, label=r'Mean')
    plt.fill_between(density_test[sat_cut:]/n0, \
                     cs2_sampled['mean']-cs2_sampled['std'], cs2_sampled['mean']+cs2_sampled['std'], \
                     zorder=12, label=r'1-$\sigma$')
    plt.fill_between(density_test[sat_cut:]/n0, \
                     cs2_sampled['mean']-1.96*cs2_sampled['std'], \
                     cs2_sampled['mean']+1.96*cs2_sampled['std'], \
                     label=r'2-$\sigma$', zorder=10)
    plt.axhline(y=1.0/3.0, linestyle='dashed')
    plt.xlabel(r'$n/n_{0}$')
    plt.ylabel(r'$c_{s}^{2}(n)$')
    plt.xlim(0.13/n0, 16.4/n0)
    plt.xscale('log')
    plt.ylim(0.0, 0.75)
    plt.legend(loc='lower right')
    plt.show()

    return None


def boundary_conditions(dens, pres_dict, index=0):
    
    '''
    Helper function to find boundary conditions
    from the pQCD results. 
    
    Parameters:
    -----------
    dens : numpy.ndarray
        The density array as input to find the BCs.
    
    pres_dict : dict
        The dictionary of pressure values 
        corresponding to the input density array.
    
    Returns:
    --------
    mu_FG : numpy.ndarray
        The 1-d array of chemical potentials
        corresponding to the values of density that
        were input. 
        
    mU_FG : numpy.ndarray
        The array of shape [:,None] that is
        used in the gsum truncation error
        analysis. 
        
    edens_dict : dict
        The energy density values at the chosen
        density index, used as the BCs for the 
        speed of sound calculation.
    
    '''
    
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


def pal_eos(kf):
    
    '''
    Python version of PAL (Prakash, Ainsworth, Lattimer) EOS. 
    Coupling constants found via the FORTRAN code paleoscc.f90,
    not included in this function. This function is designed
    to be used as a mean function in the GP for chiral EFT.
    
    Parameters:
    -----------
    kf : numpy.ndarray
        The Fermi momentum to be used to calculate PAL for
        the energy per particle.
        
    Returns:
    --------
    enperpart_kf : numpy.ndarray
        The energy per particle, in terms of the
        Fermi momentum.
    '''
    
    # extract coupling constants from cc dict
    K0 = 260. #cc['K0']       # MeV
    A = -47.83618 #cc['A']    # MeV
    B = 31.01158 #cc['B']     # MeV
    Bp = 0. #cc['Bp']       
    Sig = 1.500259  #cc['Sig'] 
    
    # if we want Bp != 0.0
#     A = -22.97032
#     B = 22.3410432
#     Bp = 0.2
#     Sig = 1.9999723
    
    # other constants
    hc = 197.33     # MeV fm
    n0 = 0.164      # fm^-3
    mn = 939.       # MeV
    kf0 = (1.5*np.pi**2.*0.164)**(1./3.)    # fm^1
    ef0 = (hc*kf0)**2./2./939.              # MeV
    sufac = (2.**(2./3.)-1.)*0.6*ef0        # MeV
    s0 = 30.                                # MeV
    
    # other coupling constants
    C1 = -83.841                            # MeV
    C2 = 22.999                             # MeV
    Lambda1 = 1.5*kf0                       # fm^-1
    Lambda2 = 3.*kf0                        # fm^-1
    
    # conversion from kf to n to solve that problem
    n = 2.0 * kf**3.0 / (3.0 * np.pi**2.0)          # fm^-3
    
    # write it as E/A first
    one = mn * n0 * (n/n0) + (3.0/5.0)*ef0*n0*(n/n0)**(5.0/3.0)       # MeV/fm^3
    two = 0.5*A*n0*(n/n0)**2.0 + (B*n0*(n/n0)**(Sig+1.0))/(1.0 + Bp * (n/n0)**(Sig - 1.0))   # MeV/fm^3
    sum_1 = C1 * (Lambda1/kf0)**3.0 * ((Lambda1/kf) - np.arctan(Lambda1/kf))  # MeV
    sum_2 = C2 * (Lambda2/kf0)**3.0 * ((Lambda2/kf) - np.arctan(Lambda2/kf))  # MeV 
    three = 3.0 * n0 * (n/n0) * (sum_1 + sum_2)                               # MeV/fm^3
    
    eps_kf = one + two + three     # MeV/fm^3
    
    # convert to E/A from eps
    enperpart_kf = eps_kf * n    # MeV
    
    # now calculate pressure using differentiation
 #   derivEA = np.gradient(enperpart_kf, kf)
 #   pressure_kf = (n * kf / 3.0) * np.asarray(derivEA)
    
    return enperpart_kf

def pressure_pal_eos(kf):
    
    '''
    The PAL EOS pressure calculation.
    
    Parameters:
    -----------
    kf : numpy.ndarray
        The Fermi momentum.
    
    Returns:
    --------
    pressure_kf : numpy.ndarray
        The pressure in terms of the Fermi
        momentum.
    '''
    
    # calculate n first again (so we don't have to pass it in)
    n = 2.0 * kf**3.0 / (3.0 * np.pi**2.0)
    
    # now calculate pressure using differentiation
    derivEA = ndt.Derivative(pal_eos, step=1e-4, method='central')
    pressure_kf = (n * kf / 3.0) * np.asarray(derivEA(kf))
    
    return pressure_kf

def get_closest_mask(array, values):
    """Returns a mask corresponding to the locations in array that are closest to values.
    
    array and values must be sorted
    
    Taken from gsum, originally written by J. A. Melendez
    """
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) \
                                              < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1
    return np.isin(np.arange(len(array)), idxs)

def get_linear_mask_in_log_space(x, x_min, x_max, log_x_step, base=np.e):
    
    '''
    Mask for getting linear data in the log space. Written by
    J. A. Melendez.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Array of x values.
        
    x_min : float
        Min x value.
    
    x_max : float
        Max x value.
        
    log_x_step : float
        The step size.
        
    base : float
        The base of the log we are using. Default
        is natural log (np.e). 
        
    Returns:
    --------
    The linear mask in the logarithmic space.
    
    '''
    lin_x = np.arange(
        np.emath.logn(n=base, x=x_min),
        np.emath.logn(n=base, x=x_max),
        log_x_step
    )
    closest = get_closest_mask(np.emath.logn(n=base, x=x), lin_x)
    return (x <= x_max) & (x >= x_min) & closest