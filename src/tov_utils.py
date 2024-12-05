import numpy as np

# create a routine for the TOV sequence
def tov_data(edens_full, pres_dict, save=False, filepath=None):

    # number of samples
    samples = len(pres_dict['samples'].T)

    # organize the samples into the correct pairing (same style as a file would be)
    low_den_file = np.loadtxt("../data/NSM_data/MFT_ns6p.dat", skiprows=1)

    # turn edens_full into an array
    edens = np.asarray(edens_full).T

    # save data in dat file backwards (without cs2 to start)
    tov_index = (np.where([pres_dict['dens'][i] <= 0.08 for i in range(len(pres_dict['dens']))])[0][-1] + 1)
    edens_final = edens[tov_index:]
    gp_final = np.asarray([pres_dict['samples'][tov_index:, i] for i in range(samples)]).T #*\
#                             convert_interp(pres_dict['dens'][tov_index:]) for i in range(samples)]).T
    density_final = pres_dict['dens'][tov_index:]

    # run through and append the low density data to these arrays and then save to file
    edens_tov = np.asarray([np.concatenate((low_den_file[::-1,0], edens_final[:,i]))\
                            for i in range(samples)]).T
    pres_tov = np.asarray([np.concatenate((low_den_file[::-1,1], gp_final[:,i])) \
                            for i in range(samples)]).T
    dens_tov = np.concatenate((low_den_file[::-1,2], density_final)).reshape(-1,1)
    cs2_tov = np.zeros(len(density_final) + len(low_den_file[:,0]))

    # save the result if desired
    if save is True:
        np.savez(filepath, density=dens_tov, \
                edens=edens_tov, pres=pres_tov, cs2=cs2_tov)
    else:
        print('Data not saved.')

    # make a dict out of the results and return
    tov_dict = {
        'dens': dens_tov,
        'edens': edens_tov,
        'pres': pres_tov,
        'cs2': cs2_tov
    }

    return tov_dict

def causality_stability(cs2, edens, pressure):
    
    # check if reshaping is needed (probably unnecessary)
    if cs2.ndim == 1:
        cs2 = cs2.reshape(-1,1)

    # run over the draws (must be given in [density, draws]!)
    ind_list = []
    for i in range(len(cs2)):
        for j in range(len(cs2.T)):
            if cs2[i,j] > 1.0 or cs2[i,j] < 0.0:
                ind_list.append(j)   # cut out the draws (make sure this works right)
                
    # cut these out of the draws in the TOV results below
    cs2_reduced = np.delete(cs2, ind_list, axis=1)
    pressure_reduced = np.delete(pressure, ind_list, axis=1)
    edens_reduced = np.delete(edens, ind_list, axis=1)

    return cs2_reduced, edens_reduced, pressure_reduced
