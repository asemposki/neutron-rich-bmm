### Chiral EFT class implementing the
### code from C. Drischler, J. A. Melendez,
### R. J. Furnstahl, and D. R. Phillips in their
### github.com/buqeye/nuclear-matter-convergence
### repo from their 2021 PRC and PRL papers.

### Class written by : Alexandra Semposki
### Last edited : 16 May 2023

# import packages and paths
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
import matplotlib.patches as mpatches
import pandas as pd
import time
import corner
import gptools

import sys
sys.path.append('../nuclear-matter-convergence/nuclear_matter/')

from matter import fermi_momentum, nuclear_density
from matter import compute_pressure, compute_pressure_cov
from matter import compute_slope, compute_slope_cov
from matter import compute_compressibility, compute_compressibility_cov
from matter import compute_speed_of_sound
from graphs import setup_rc_params
from utils import InputData
from graphs import confidence_ellipse, confidence_ellipse_mean_cov
from graphs import add_top_order_legend, compute_filled_handles, plot_empirical_saturation
from matter import kf_derivative_wrt_density

# from our version of this code
# from derivatives_new import ObservableContainer, SymmetryEnergyContainer
from derivatives_new import ObservableContainer, SymmetryEnergyContainer

# begin the class
class Chiral:

    def __init__(self, density_input=None, Lambda=500, high_density=True):

        '''
        The class for the chiral EFT EOS model. Will be used
        in the Taweret model class for the chiral EOS. This init
        function imports the necessary data to begin the calculation, and
        sets up the density and data for interpolation. Choice of Lambda 
        (450, 500) is also made here. 

        Parameters:
        -----------
        density_input : numpy.linspace
            The density regime for the chiral EOS. Default is set below 
            in the code. 

        Lambda : int
            The desired cutoff to use. Default is 500; either
            450 or 500 MeV may currently be chosen. 

        high_density : bool 
            Decision to use high density data or not.
            Default is True. 

        Returns:
        --------
        None.
        '''

        # set cutoff
        self.Lambda = Lambda

        if high_density is True:
            filename = '../nuclear-matter-convergence/data/all_matter_data_high_density.csv'
        else:
            filename = '../nuclear-matter-convergence/data/all_matter_data.csv'

        # import the data
        data = InputData(filename, Lambda)

        # set orders and breakdown scale
        self.orders = np.array([0, 2, 3, 4])  
        self.breakdown = 600 # MeV  

        # set up all of the necessary parameters
        self.kf_n = data.kf_n
        self.Kf_n = data.Kf_n
        self.kf_s = data.kf_s
        self.Kf_s = data.Kf_s
        self.kf_d = data.kf_avg
        self.Kf_d = data.Kf_avg

        self.density = data.density

        self.ref_2bf = data.ref_2bf
        self.ref_n_3bf = data.ref_n_3bf
        self.ref_s_3bf = data.ref_s_3bf
        self.ref_d_3bf = data.ref_avg_3bf

        self.y_s_2_plus_3bf = data.y_s_2_plus_3bf
        self.y_n_2_plus_3bf = data.y_n_2_plus_3bf
        self.y_d_2_plus_3bf = data.y_d_2_plus_3bf

        self.y_s_2bf = data.y_s_2bf
        self.y_n_2bf = data.y_n_2bf
        self.y_d_2bf = data.y_d_2bf

        self.y_s_3bf = data.y_s_3bf
        self.y_n_3bf = data.y_n_3bf
        self.y_d_3bf = data.y_d_3bf

        self.fit_n2lo = data.fit_n2lo
        self.fit_n3lo = data.fit_n3lo

        self.data = data 
    
        # set up density
        if density_input is not None:
            self.density_all = density_input
        else:
            self.density_all = np.arange(self.density[0], self.density[-1], 0.005)

        self.N_all = len(self.density_all)

        #neutron matter
        self.kf_n_all = fermi_momentum(self.density_all, degeneracy=2)
        self.Kf_n_all = self.kf_n_all[:, None]

        #symmetric matter
        self.kf_s_all = fermi_momentum(self.density_all, degeneracy=4)
        self.Kf_s_all = self.kf_s_all[:, None]

        #symmetry energy
        self.kf_d_all = (self.kf_n_all + self.kf_s_all) / 2.
        self.Kf_d_all = self.kf_d_all[:, None]

        # error band setup for MBPT calculations
        min_uncertainty = 0.02  # Twenty keV
        uncertainty_factor = 0.001  # 0.1%

        self.err_y_n = np.abs(self.y_n_2_plus_3bf[:, -1]) * uncertainty_factor
        self.err_y_n[np.abs(self.err_y_n) < min_uncertainty] = min_uncertainty

        self.err_y_s = np.abs(self.y_s_2_plus_3bf[:, -1]) * uncertainty_factor
        self.err_y_s[np.abs(self.err_y_s) < min_uncertainty] = min_uncertainty

        self.err_y_d = np.sqrt(self.err_y_n**2 + self.err_y_s**2)

        # momenta and std dev setup
        self.kf0_n = fermi_momentum(0.164, 2)
        self.kf0_s = fermi_momentum(0.164, 4)

        self.ref_neutron = 16 / self.kf0_n**2
        self.ref_nuclear = 16 / self.kf0_s**2

        if self.Lambda == 500:
            self.std_neutron = 1.00
            self.ls_neutron = 0.973
            self.std_nuclear = 2.95
            self.ls_nuclear = 0.484

        elif self.Lambda == 450:
            self.std_neutron = 0.8684060649936118
            self.ls_neutron = 0.7631421388401067
            self.std_nuclear = 2.6146499024837073
            self.ls_nuclear = 0.46603268529311087

        if self.Lambda == 450:
            self.rho = 0.95
            self.ls_n_sym = (self.ls_neutron + self.ls_nuclear) / 2
            self.ls_s_sym = None
        
        elif self.Lambda == 500:
            self.rho = None   # letting it calculate rho
            self.ls_n_sym = self.ls_neutron
            self.ls_s_sym = self.ls_nuclear

        return None
    

    def data_interpolation(self, density_int=None, kf_s_int=None, extend=False):

        '''
        The interpolation of the data using the observable
        containers from nuclear-matter-convergence for neutron matter,
        symmetric nuclear matter, and the symmetry energy calculation.

        Parameters:
        -----------
        density_int : numpy.ndarray
            The interpolated number density values from the mu(n) inversion.
            Default is None; this will leave the container to use the 
            original interpolated density_all. 

        kf_s_int : numpy.ndarray
            The interpolated Fermi momenta for SNM. This will be replaced
            when calculating the inverted density and chemical potential.
            Default is None; this will leave the container to use the
            original interpolated kf_s. 
            
        extend: bool
            Whether or not to extend the data using the mean function 
            estimation and extending the truncation error. Default is False.

        Returns:
        --------
        self.obs_neutron, self.obs_nuclear, self.obs_sym_energy : objects
            The objects that contain the important interpolation information
            for the chiral EFT EOS for PNM, SNM, and the symmetry energy. 
        '''

        setup_time_start = time.time()
        verbose = True

        # alter the density and kf_s in the obs_nuclear container
        # if the routine is called for the inversion calculation
        if density_int is not None:
            self.dens_interp = density_int
        else: 
            self.dens_interp = self.density_all 

        if kf_s_int is not None:
            self.kf_s_all = kf_s_int

        print('Setting up neutron matter...', flush=True)
        self.obs_neutron = ObservableContainer(
            density=self.density,
            kf=self.kf_n,
            y=self.y_n_2_plus_3bf,
            orders=self.orders,
            density_interp=self.density_all,
            kf_interp=self.kf_n_all,
            std=self.std_neutron,
            ls=self.ls_neutron,
            ref=self.ref_neutron,
            breakdown=self.breakdown,
            err_y=self.err_y_n,
            include_3bf=False,
            derivs=[0, 1, 2],
            verbose=verbose,
            extend=extend,
        )

        print('Setting up nuclear matter...', flush=True)
        self.obs_nuclear = ObservableContainer(
            density=self.density,
            kf=self.kf_s,
            y=self.y_s_2_plus_3bf,
            orders=self.orders,
            density_interp=self.dens_interp, 
            kf_interp=self.kf_s_all,
            std=self.std_nuclear,
            ls=self.ls_nuclear,
            ref=self.ref_nuclear,
            breakdown=self.breakdown,
            err_y=self.err_y_s,
            include_3bf=False,
            derivs=[0, 1, 2],
            verbose=verbose,
            extend=extend,
        )

        print('Setting up symmetry energy...', flush=True)
        self.obs_sym_energy = SymmetryEnergyContainer(
            density=self.density,
            y=self.y_d_2_plus_3bf,
            orders=self.orders,
            density_interp=self.density_all,
            std_n=self.std_neutron,
            ls_n=self.ls_n_sym,
            std_s=self.std_nuclear,
            ls_s=self.ls_s_sym,
            ref_n=self.ref_neutron,
            ref_s=self.ref_nuclear,
            breakdown=self.breakdown,
            err_y=self.err_y_d,
            include_3bf=False,
            derivs=[0, 1],
            verbose=verbose,
            rho=self.rho,
        )

        print('Setup time:', time.time() - setup_time_start)

        return self.obs_neutron, self.obs_nuclear, self.obs_sym_energy
    

    def energy_per_particle(self, add_rest_mass=False, case='SNM', orders='all'):

        '''
        A function that ouputs the energy per particle with truncation 
        errors from gsum. 

        Parameters:
        -----------
        add_rest_mass : bool
            If desired, will add the rest mass to the energy per
            particle. 

        orders : str
            Command to determine whether we use all orders for the EFT or
            just one of them. Default is 'all', but 'N3LO' will convert
            to only one EFT at N3LO. 

        Returns:
        --------
        self.energies_s, self.energy_s_stds : numpy.ndarray
            The energy per particle (E/A) and standard deviation of E/A.

        if add_rest_mass is True:
        self.energies_s_mn, self.energy_s_stds : numpy.ndarray
            E/A (inclusive of the rest mass) and the standard deviation 
            of E/A. 
        '''
        
        if case == 'SNM':

            # adding the rest mass in (average p,n)
            self.rest_mass = 938.91875434       # MeV

            # energy per particle
            if orders == 'all':
                self.energies_s = np.array([self.obs_nuclear.get_pred(order=n, deriv=0) \
                                    for n in self.orders]).T
                self.energy_s_stds = np.array([self.obs_nuclear.get_std(order=n, deriv=0, \
                                        include_trunc=True) for n in self.orders]).T
            elif orders == 'N3LO':
                self.energies_s = np.array([self.obs_nuclear.get_pred(order=4,\
                                             deriv=0)]).T
                self.energy_s_stds = np.array([self.obs_nuclear.get_std(order=4, \
                                             deriv=0, include_trunc=True)]).T

            # if rest_mass is True, add it to E/A
            if add_rest_mass is True:
                self.energies_s_mn = self.energies_s + self.rest_mass
                return self.energies_s_mn, self.energy_s_stds

            else:
                return self.energies_s, self.energy_s_stds
            
        elif case == 'PNM':
            
            # adding the rest mass in (n only here)
            self.rest_mass = 939.6       # MeV

            # energy per particle
            if orders == 'all':
                self.energies_n = np.array([self.obs_neutron.get_pred(order=n, deriv=0) \
                                    for n in self.orders]).T
                self.energy_n_stds = np.array([self.obs_neutron.get_std(order=n, deriv=0, \
                                        include_trunc=True) for n in self.orders]).T
            elif orders == 'N3LO':
                self.energies_n = np.array([self.obs_neutron.get_pred(order=4,\
                                             deriv=0)]).T
                self.energy_n_stds = np.array([self.obs_neutron.get_std(order=4, \
                                             deriv=0, include_trunc=True)]).T

            # if rest_mass is True, add it to E/A
            if add_rest_mass is True:
                self.energies_n_mn = self.energies_n + self.rest_mass
                return self.energies_n_mn, self.energy_n_stds

            else:
                return self.energies_n, self.energy_n_stds
            


    def pressure(self, orders='all', matter='SNM'):

        '''
        The pressure of the chiral EOS. 
        Note: Rest mass will not affect this calculation, so no option for 
        adding the rest mass is included here. 

        Parameters:
        -----------
        orders : str
            Command to determine how many orders in the EFT expansion
            to calculate. Default is 'all', but can be set to 'N3LO' to
            only return that one.

        Returns:
        --------
        self.pressure_s, self.pressure_s_stds : numpy.ndarray
            The pressure and standard deviation of the pressure. 
        '''

        if matter == 'SNM':
            # set up the pressure and std dev
            pressures_s = []
            pressure_s_stds = []
            pressure_s_cov_n = {}
            pressure_s_cov_arr = np.zeros([len(self.obs_nuclear.kf_interp), len(self.obs_nuclear.kf_interp), len(self.orders)])

            if orders == 'all':

                for i, n in enumerate(self.orders):
                    pressure_s = compute_pressure(
                        self.obs_nuclear.density_interp,
                        self.obs_nuclear.kf_interp,
                        dE=self.obs_nuclear.get_pred(order=n, deriv=1)
                    )
                    pressure_s_cov = compute_pressure_cov(
                        self.obs_nuclear.density_interp,
                        self.obs_nuclear.kf_interp,
                        dE_cov=self.obs_nuclear.get_cov(order=n, deriv1=1, deriv2=1)
                    )

                    pressure_s_cov_n[n] = pressure_s_cov

                    pressure_s_std = np.sqrt(np.diag(pressure_s_cov))

                    pressures_s.append(pressure_s)
                    pressure_s_stds.append(pressure_s_std)

                pressures_s = np.array(pressures_s).T
                pressure_s_stds = np.array(pressure_s_stds).T

                for i,value in zip(range(len(self.orders)), pressure_s_cov_n.values()):
                    pressure_s_cov_arr[:,:,i] = np.array(value).T

                # transform into class variables
                self.pressures_s = pressures_s 
                self.pressure_s_stds = pressure_s_stds 
                self.pressure_s_cov = pressure_s_cov_arr

                return self.pressures_s, self.pressure_s_stds, self.pressure_s_cov 

            elif orders == 'N3LO':

                pressures_s = compute_pressure(
                    self.obs_nuclear.density_interp,
                    self.obs_nuclear.kf_interp,
                    dE=self.obs_nuclear.get_pred(order=4, deriv=1)
                )

                pressure_s_cov = compute_pressure_cov(
                    self.obs_nuclear.density_interp,
                    self.obs_nuclear.kf_interp,
                    dE_cov=self.obs_nuclear.get_cov(order=4, deriv1=1, deriv2=1)
                )

                pressure_s_stds = np.sqrt(np.diag(pressure_s_cov))

                pressures_s = np.array(pressures_s).T
                pressure_s_stds = np.array(pressure_s_stds).T
                pressure_s_cov = np.array(pressure_s_cov).T

                # transform into class variables
                self.pressures_s = pressures_s 
                self.pressure_s_stds = pressure_s_stds 
                self.pressure_s_cov = pressure_s_cov

                return self.pressures_s, self.pressure_s_stds, self.pressure_s_cov 

        if matter == 'PNM':
            # set up the pressure and std dev
            pressures_n = []
            pressure_n_stds = []
            pressure_n_cov_n = {}
            pressure_n_cov_arr = np.zeros([len(self.obs_neutron.kf_interp), len(self.obs_neutron.kf_interp), len(self.orders)])

            if orders == 'all':

                for i, n in enumerate(self.orders):
                    pressure_n = compute_pressure(
                        self.obs_neutron.density_interp,
                        self.obs_neutron.kf_interp,
                        dE=self.obs_neutron.get_pred(order=n, deriv=1)
                    )
                    pressure_n_cov = compute_pressure_cov(
                        self.obs_neutron.density_interp,
                        self.obs_neutron.kf_interp,
                        dE_cov=self.obs_neutron.get_cov(order=n, deriv1=1, deriv2=1)
                    )

                    pressure_n_cov_n[n] = pressure_n_cov

                    pressure_n_std = np.sqrt(np.diag(pressure_n_cov))

                    pressures_n.append(pressure_n)
                    pressure_n_stds.append(pressure_n_std)

                pressures_n = np.array(pressures_n).T
                pressure_n_stds = np.array(pressure_n_stds).T

                for i,value in zip(range(len(self.orders)), pressure_n_cov_n.values()):
                    pressure_n_cov_arr[:,:,i] = np.array(value).T

                # transform into class variables
                self.pressures_n = pressures_n 
                self.pressure_n_stds = pressure_n_stds 
                self.pressure_n_cov = pressure_n_cov_arr

                return self.pressures_n, self.pressure_n_stds, self.pressure_n_cov 
    

    def energy_dens(self, add_rest_mass=False, orders='all'):

        '''
        Computes the energy density and standard deviation of the energy
        density of the chiral EOS. Adds rest mass if desired. 

        Parameters:
        -----------
        add_rest_mass : bool
            The option to include the rest mass in the E/A (and in the 
            energy density). 

        orders : str
            Option to either calculate all orders ('all') or only N3LO
            ('N3LO'). Default is 'all'.

        Returns:
        --------
        self.energy_density, self.energy_density_s_stds : numpy.ndarray
            The energy density and standard deviation of the energy density.
        '''

        # if rest_mass is true, add it and compute
        if add_rest_mass is True:

            if orders == 'all':
                # call energy per particle function
                self.energies_s_mn, self.energy_s_stds = \
                    self.energy_per_particle(add_rest_mass=True, orders='all')
                
                self.energy_density_s = np.zeros([len(self.energies_s_mn), 4])
                self.energy_density_s_stds = np.zeros([len(self.energy_s_stds), 4])

                for i in range(4):
                    self.energy_density_s[:,i] = self.energies_s_mn[:,i] \
                        * self.density_all 
                    self.energy_density_s_stds[:,i] = \
                        self.energy_s_stds[:,i] * self.density_all 
            
            elif orders == 'N3LO':

                # call energy per particle function
                self.energies_s_mn, self.energy_s_stds = \
                    self.energy_per_particle(add_rest_mass=True, orders='N3LO')

                self.energy_density_s = self.energies_s_mn[:,0] * self.dens_interp
                self.energy_density_s_stds = self.energy_s_stds[:,0] * self.dens_interp

        # otherwise calculate eps and std dev as normal
        else:

            if orders == 'all':

                self.energies_s, self.energy_s_stds = \
                    self.energy_per_particle(add_rest_mass=False, orders='all')
                
                self.energy_density_s = np.zeros([len(self.energies_s), 4])
                self.energy_density_s_stds = np.zeros([len(self.energy_s_stds), 4])

                for i in range(4):
                    self.energy_density_s[:,i] = self.energies_s[:,i] \
                        * self.density_all 
                    self.energy_density_s_stds[:,i] = self.energy_s_stds[:,i] \
                        * self.density_all 
                
            elif orders == 'N3LO':

                # call energy per particle function
                self.energies_s, self.energy_s_stds = \
                    self.energy_per_particle(add_rest_mass=False, orders='N3LO')

                self.energy_density_s = self.energies_s[:,0] * self.dens_interp
                self.energy_density_s_stds = self.energy_s_stds[:,0] * self.dens_interp

        return self.energy_density_s, self.energy_density_s_stds
    

    def chemical_potential(self, method=1, add_rest_mass=False):
        
        '''
        Calculation of the chemical potential depending on which
        type is desired: either type=1 ((P+eps)/n) or type=2 (deps/dn).
        Both should be equivalent if thermodynamic consistency is 
        preserved. 

        Parameters:
        ------------
        method : int
            The type of calculation to obtain the chemical potential. 
            1 : (P+eps)/n
            2 : d(eps)/dn
            Default is 1. 

        add_rest_mass : bool
            Adds the rest mass to the energy per particle if desired. 
            Default is False. 
        
        Returns:
        --------
        self.mu_s, self.mu_s_stds : numpy.ndarray
            The chemical potential and std dev for the chiral EOS, wrt n. 
        '''

        # call the energy_density function and tell it rest mass choice
        self.energy_density_s, self.energy_density_s_stds = \
            self.energy_dens(add_rest_mass=add_rest_mass)

        # type 1
        if method == 1:

            # call the pressure function
            self.pressures_s, self.pressure_s_stds, self.pressure_s_cov = self.pressure()

            ti_upper = self.pressures_s + self.energy_density_s
            ti_upper_stds = self.pressure_s_stds + self.energy_density_s_stds
            mu_s = np.zeros([len(ti_upper), 4])
            mu_s_stds = np.zeros([len(ti_upper), 4])

            for i in range(4):
                mu_s[:,i] = ti_upper[:,i]/self.density_all
                mu_s_stds[:,i] = ti_upper_stds[:,i]/self.density_all

            # class variable
            self.mu_s = mu_s 
            self.mu_s_stds = mu_s_stds

            return self.mu_s, self.mu_s_stds

        elif method == 2:

            # compute the derivative of the energy density
            k_F = (3.0/2.0 * np.pi**2.0 * self.density_all)**(1.0/3.0)

            dE = np.zeros([len(self.density_all), 4])
            dE_cov = np.zeros([len(dE), len(dE)])
            dE_std = np.zeros([len(dE), 4])

            mu_s_deriv = np.zeros([len(self.density_all), 4])
            mu_s_deriv_stds = np.zeros([len(self.density_all), 4])

            for i, n in enumerate(self.orders):
                dE[:,i] = self.obs_nuclear.get_pred(order=n, deriv=1) 
                dE_cov = self.obs_nuclear.get_cov(order=n, deriv1=1, deriv2=1)
                dE_std[:,i] = np.sqrt(np.diag(dE_cov))
                
            for i in range(4):
                mu_s_deriv[:,i] = (k_F * dE[:,i])/3.0 + \
                    (self.energy_density_s[:,i]/self.density_all)
                mu_s_deriv_stds[:,i] = (k_F * dE_std[:,i])/3.0 + \
                      (self.energy_density_s_stds[:,i]/self.density_all)
                
            # class variable
            self.mu_s = mu_s_deriv
            self.mu_s_stds = mu_s_deriv_stds

            return self.mu_s, self.mu_s_stds

        else:
            raise ValueError('Method must be either 1 or 2.')


    def inversion(self, guess=0.33):

        '''
        *** Need clever way to rewrite this inversion scheme for the 
        input space we'll need to implement the BMM process. *** 

        Inverts the function mu(n) in favour of n(mu). For Lambda = 500 MeV,
        able to use the fsolve function to do this. For Lambda = 450 MeV,
        must manually invert. ***Note: because of this, the range of mu is 
        not the same for both cases of Lambda. 

        Parameters:
        -----------
        guess : float
            The guess for the fsolve function to invert for Lambda = 500 MeV.
            If using Lambda = 450 MeV, this argument is ignored. 
            Default is 0.33. 

        Returns:
        --------
        self.density_mu_N3LO : numpy.ndarray
            The array in density that we obtain from inversion. 

        self.mu_array_N3LO : numpy.ndarray
            The inverted chemical potential array. 

        self.kf_N3LO : numpy.ndarray
            The new array in kf for the observable container once mu has been 
            inverted. 
        '''

        # set interpolation space outside of cutoff dependence
        nnew = np.linspace(0.05, 0.339, 500)   # should this be changed to 16?
        #nnew = np.linspace(min(self.density_all), max(self.density_all), 500)
        
        if self.Lambda == 500:

            # N3LO results
            mu_inversion_N3LO = np.asarray(self.invert(self.density_all, self.mu_s[:,3], guess=guess, nnew=nnew))

            # send in the densities in terms of the mu we want
            self.mu_N3LO_array = mu_inversion_N3LO[0] # mu values we now are working with in ALL cases below
            self.density_mu_N3LO = mu_inversion_N3LO[1] # densities found given a specific mu array

            # figure out the density cutoff in mu by finding nearest value
            n_cutoff = mu_inversion_N3LO[1][0]
            nearest = self.density_mu_N3LO.flat[np.abs(self.density_mu_N3LO - n_cutoff).argmin()]
            index = np.where(self.density_mu_N3LO==nearest)[0][0]
            mu_cutoff = self.mu_N3LO_array[index]

            # now use this cutoff to stop from calculating beyond this point 
            index = np.where(self.mu_N3LO_array > mu_cutoff)
            self.mu_N3LO_array = self.mu_N3LO_array[index[0][0]:]
            self.density_mu_N3LO = self.density_mu_N3LO[index[0][0]:]
            
            # print the range in density to check
            print(min(self.density_mu_N3LO), max(self.density_mu_N3LO))

            # also recalculate kf for the observable container
            self.kf_N3LO = (3.0 * np.square(np.pi) * self.density_mu_N3LO / 2.0)**(1.0/3.0)
            
            return self.density_mu_N3LO, self.mu_N3LO_array, self.kf_N3LO

        elif self.Lambda == 450:

            # try just inverting the interpolation
            mu_N3LO = self.mu_s[:,3]

            # interpolate the array in mu
            mu_N3LO_interp = interpolate.interp1d(self.density_all, mu_N3LO, kind='cubic')
            
            # set a new density linspace and mu for tight interpolation
            mu_new = mu_N3LO_interp(nnew)

            # determine region where two solutions exist to avoid it
            mu_min = min(mu_new)
            mu_max = mu_new[0]
            index_mu = np.where(mu_new==mu_min)[0][0]
            
            # write new density array without the bottom half of mu
            mu_upper = mu_new[index_mu:]
            n_upper = nnew[index_mu:]
            
            # use interpolation to get more points in density (function)
            n_interp = interpolate.interp1d(mu_upper, n_upper, kind='cubic')
            
            # use the function to generate density curve in mu
            nearest = mu_N3LO.flat[np.abs(mu_N3LO - mu_max).argmin()]
            index = np.where(mu_N3LO==nearest)[0][0]  
            
            # chemical potential above density cutoff
            mu_slice = mu_N3LO[index+1:]
            
            # cut this array to fit the interpolation length
            nearest = mu_slice.flat[np.abs(mu_slice - mu_upper[-1]).argmin()]
            index = np.where(mu_slice==nearest)[0][0]  
            mu_slice = mu_slice[:index-1]
            
            # new density array from interpolant and mu values
            nmu_new = n_interp(mu_slice)

            # collect arrays from before
            self.mu_N3LO_array = mu_slice
            self.density_mu_N3LO = nmu_new

            # print the density limits now
            print(min(self.density_mu_N3LO), max(self.density_mu_N3LO))

            # also recalculate kf for the observable container
            self.kf_N3LO = (3.0 * np.square(np.pi) * self.density_mu_N3LO / 2.0)**(1.0/3.0)

            return self.density_mu_N3LO, self.mu_N3LO_array, self.kf_N3LO
        

    def invert(self, n, mu, guess=0.33, nnew=None):

        '''
        Inversion function that uses fsolve. At present, only used
        for Lambda = 500 MeV. 

        Parameters:
        -----------
        n : numpy.ndarray
            The array of densities that we currently possess.
        
        mu : numpy.ndarray
            The array of chemical potential values for the range
            of densities in n.

        guess : float
            The guess for fsolve. Default is 0.33.

        nnew : numpy.ndarray
            The values for the new array in density. Default is None.

        Returns:
        --------
        mu_new : numpy.ndarray
            Array of mu results at new points in density. 
            
        f_n_result : numpy.ndarray
            The results of the root finding wrt the new
            chemical potential array.

        '''
    
        # interpolate the arrays
        mu_interp = interpolate.interp1d(n, mu, kind='cubic')
        
        # set a new density linspace and mu for tight interpolation
        if nnew is not None:
            mu_new = mu_interp(nnew)
        else:
            mu_new = mu_interp(n)
        
        # call root finder
        f_n_result = []
        for i in mu_new:
            f_n_result.append(self.f_n(i, mu_interp, guess=guess))
        
        return mu_new, f_n_result
    

    def f_n(self, mu, mu_func, guess=0.33):

        '''
        fsolve function to invert mu(n) to n(mu).

        Parameters:
        -----------
        mu : float
            The current value in the chemical potential to be 
            inverted.

        mu_func : function
            The function from interp1d that is to be solved for n.

        guess : float
            The guess for fsolve to find a root. Default is 0.33. 

        Returns:
        --------
            The value of n with respect to the given values of the 
            chemical potential. 
        '''

        return optimize.fsolve(lambda n : mu - mu_func(n), x0 = guess)[0]