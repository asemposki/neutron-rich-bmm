### Chiral EFT EOS model for Taweret
### Written by : Alexandra Semposki
### Date last edited : 11 May 2023

# NOTE: we want the N3LO results from this code only, so we are
# only sending those back; can change if needed

# This model class is not currently used in the paper,
# but is here for use if one wishes to do so.

import numpy as np
import sys 

# imports from chiral EOS class
from chiral import Chiral

# Taweret import
sys.path.append('../../Taweret')

from Taweret.core.base_model import BaseModel

# write the class 
class Chiral_model(BaseModel):

    def __init__(self, density=None, Lambda=500, high_density=True):

        '''
        Model class to give Taweret to mix. 

        Parameters:
        -----------
        density : numpy.array
            The density specified for calculation of the chiral
            EOS. Default is None, which will make the chiral EOS
            default to the original data density range. 
            
        Lambda : int
            The value of the cutoff chosen. Can be either
            450 MeV or 500 MeV. 

        high_density : bool 
            Whether we want to use high-density data or not.
            Default is True. Sets up the data for interpolation
            in the chiral EOS class. 
        
        Returns:
        --------
        None.
        '''

        # set class variables
        self.Lambda = Lambda
        self.density = density

        # call Chiral class and instantiate
        self.chiral = Chiral(density_input=self.density, Lambda=self.Lambda, \
                             high_density=high_density) 

        return None 
    

    def evaluate(self, input_space=None, N3LO=True, scaled=True, extend=False):

        '''
        Returns the mean and standard deviation of the chiral EFT EOS
        in terms of pressure wrt baryon chemical potential.

        Mean: calculated from the chiral EFT EOS formalism of
              C. Drischler et al. (2021). 

        Standard deviation: calculated via the truncation error
                            models in the gsum package, used in
                            the chiral EOS paper.

        Parameters:
        -----------
        input_space : numpy.array
            The input space array. Not actually necessary for this 
            function but necessary for Taweret. 
            
        N3LO : bool
            If True, returns only the N3LO results for mean and std_dev.
            Otherwise will return all results up to and through N3LO.
            
        scaled : bool
            If the data is scaled, then this is True. Else, it is False.
            Default is True.
            
        extend : bool
            Extends the data to higher truncation values. Default is
            False.

        Returns:
        --------
        mean, std_dev : numpy.ndarray
            The mean and standard deviation of the pressure. 
        '''
        
        # correct for input_space (for now)
        if input_space is not None:
            input_space = None

        # set up the data containers for interpolation
        self.obs_neutron, self.obs_nuclear, self.obs_sym_energy = \
            self.chiral.data_interpolation(density_int=None, kf_s_int=None, extend=extend)
        
        # energy per particle
        self.energies_s, self.energy_s_stds = \
            self.chiral.energy_per_particle(add_rest_mass=True, orders='all')

        # chemical potential (calls pressure and eps already)
        self.mu_s, self.mu_s_stds = \
            self.chiral.chemical_potential(method=1, add_rest_mass=True)
        
        # inversion for n(mu)
#         self.density_mu_N3LO, self.mu_N3LO_array, self.kf_N3LO = \
#             self.chiral.inversion()
        
        # call the container again with n(mu) and kf(n(mu))
#         self.obs_neutron, self.obs_nuclear, self.obs_sym_energy = \
#             self.chiral.data_interpolation(density_int=self.density_mu_N3LO, \
#                                            kf_s_int=self.kf_N3LO)
 
        # calculate pressure
        self.pressures_s, self.pressure_s_stds, self.pressure_s_cov = self.chiral.pressure(\
            orders='all')
        
        # N3LO results only
        if N3LO is True:
            mean = self.pressures_s[:,3]
            std_dev = self.pressure_s_stds[:,3]
            
            if scaled is True:
                pressure_free = np.zeros([len(self.density), 4])
                hbarc = 197.327 # MeV fm
                for i in range(4):
                    pressure_free[:,i] = ((1.0/(2.0*np.square(np.pi)))*(self.chiral.kf_s_all**4.0))*(hbarc)
                mean = mean/pressure_free[:,3]
                std_dev = std_dev/pressure_free[:,3]
                
            else:
                mean = mean
                std_dev = std_dev
            
#             for i in range(len(mean)):
#                 if mean[i] <= mean[i-1]:
#                     std_dev[i] = 1e10
        
        else:
            mean = self.pressures_s
            std_dev = self.pressure_s_stds

        return mean, std_dev
    

    def log_likelihood_elementwise(self):

        '''
        Calculates the log likelihood for the model.
        Not needed for this model.
        '''

        return None


    def set_prior(self):

        '''
        Sets the prior on the model. Not needed for
        this model. 
        '''

        return None