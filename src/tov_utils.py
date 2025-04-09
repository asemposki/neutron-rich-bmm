import numpy as np

def tov_data(edens_full, pres_dict, cs2_data=None, save=False, filepath=None):
    
    r"""
    Function to collect the data needed for the TOV input and to organize
    it such that it can be read correctly by the TOV solver we are using. 
    It also attaches the low-density EOS information up to 0.5n0 from Bethe,
    Pethick, and Sutherland, and Negele and Vautherin's work. 
    
    Example:
        tov_data(edens_full=array, pres_dict=dict, cs2_data=array)
    
    Parameters:
        edens_full (array): The array of energy densities in MeV/fm^3.
        
        pres_dict (dict): The dictionary of pressures and their standard deviations
            in MeV/fm^3, as well as the number density in fm^-3.
        
        cs2_data (array): The array of corresponding speed of sound values in 
            dimensionless units. This is optional, as these values are only 
            needed if the TOV solver is selected to calculate tidal deformability.
        
        save (bool): If the user wishes to save the data file, supply 'True'.
        
        filepath (str): If the user wishes to save the data file, supply a proper
            file path, including any external folders.
    
    Returns:
        tov_dict (dict): The full dictionary of dens, edens, pres, and cs2 
            (optionally) to be used in the TOV solver. 
    """

    # number of samples
    samples = len(pres_dict['samples'].T)

    # organize the samples into the correct pairing (same style as a file would be)
    low_den_file = np.loadtxt("../data/NSM_data/MFT_ns6p.dat", skiprows=1)

    # turn edens_full into an array  [density, draws] is needed
#     edens = np.asarray(edens_full)  #.T
#     cs2 = np.asarray(cs2_data['samples'])  #.T

    # rename dicts
    edens = edens_full
    cs2 = cs2_data

    # get organized data
    tov_index = (np.where([pres_dict['dens'][i] <= 0.08 for i in range(len(pres_dict['dens']))])[0][-1] + 1)
    edens_final = edens[tov_index:, :]
    gp_final = np.asarray([pres_dict['samples'][tov_index:, i] for i in range(samples)]).T
    density_final = pres_dict['dens'][tov_index:]
        
    if cs2_data is not None:
        cs2_final = cs2[tov_index:, :]

    # run through and append the low density data to these arrays and then save to file
    edens_tov = np.asarray([np.concatenate((low_den_file[::-1,0], edens_final[:,i]))\
                            for i in range(samples)]).T
    pres_tov = np.asarray([np.concatenate((low_den_file[::-1,1], gp_final[:,i])) \
                            for i in range(samples)]).T
    dens_tov = np.concatenate((low_den_file[::-1,2], density_final)).reshape(-1,1)
    
    if cs2_data is not None:
        # solve for the speed of sound using edens and pres at low densities
        dpdeps = np.gradient(low_den_file[::-1,1], low_den_file[::-1,0], edge_order=2)  # might need tweaking
        cs2_tov = np.asarray([np.concatenate((dpdeps, cs2_final[:,i])) for i in range(samples)]).T
    else:
        cs2_tov = np.zeros(len(density_final) + len(low_den_file[:,0]))

    # eliminate the information above the core of the star (assuming 10 times saturation high enough)
    index_10 = np.where([dens_tov[i] <= 1.64 for i in range(len(dens_tov))])[0][-1] + 1
    
    # cut the arrays
    dens_end = dens_tov  #[:index_10]
    edens_end = edens_tov  #[:index_10]
    pres_end = pres_tov  #[:index_10]
    cs2_end = cs2_tov  #[:index_10]

    # save the result if desired
    if save is True:
        np.savez(filepath, density=dens_end, \
                edens=edens_end, pres=pres_end, cs2=cs2_end)
    else: 
        print('Data not saved.')

    # make a dict out of the results and return
    tov_dict = {
        'dens': dens_end,
        'edens': edens_end,
        'pres': pres_end,
        'cs2': cs2_end
    }
    
    print('''I'm done!''')

    return tov_dict

def causality_stability(cs2, edens, pressure):
    
    r"""
    Function to eliminate any draws from the EOS that are unstable
    or acausal.
    
    Example:
        causality_stability(cs2=array, edens=array, pressure=array)
        
    Parameters:
        cs2 (array): The array of speed of sound draws. Shape should
            be [density, draw].
        
        edens (array): The array of corresponding energy densities,
            in the shape [density, draw].
        
        pressure (array): The array of corresponding pressures in the
            shape of [density, draw]. 
    
    Returns:
        cs2_reduced (array): The 2D array of reduced samples for 
            the speed of sound.
        
        edens_reduced (array): The 2D array of reduced samples for 
            the energy density.
        
        pres_reduced (array): The 2D array of reduced samples 
            for the pressure.
    """
    
    # check if reshaping is needed (probably unnecessary)
    if cs2.ndim == 1:
        cs2 = cs2.reshape(-1,1)

    # run over the draws (must be given in [density, draws]!)
    ind_list = []
    for i in range(len(cs2)):
        for j in range(len(cs2.T)):
            if cs2[i,j] > 1.0 or cs2[i,j] < 0.0:   # could do max > 1.0 or min < 0.0 or something
                ind_list.append(j)   # cut out the draws (make sure this works right)
                
    # cut these out of the draws in the TOV results below
    cs2_reduced = np.delete(cs2, ind_list, axis=1)
    pressure_reduced = np.delete(pressure, ind_list, axis=1)
    edens_reduced = np.delete(edens, ind_list, axis=1)

    return cs2_reduced, edens_reduced, pressure_reduced
