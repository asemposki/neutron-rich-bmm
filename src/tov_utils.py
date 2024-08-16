import numpy as np


# create a routine for the TOV sequence
def tov_data(edens_full, pres_dict, save=False):

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
        np.savez("../../../FASTAR/TOV/Alexandra_TOV/eos_tov_true.npz", density=dens_tov, \
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