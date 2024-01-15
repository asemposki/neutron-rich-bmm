###########################################################
# Gorda pQCD model for Taweret implementation
# Author : Alexandra Semposki
# Date : 03 August 2023
###########################################################

# imports
import numpy as np
import gsum as gm
from pqcd_reworked import PQCD
from truncation_error import Truncation
from Taweret.core.base_model import BaseModel

# class
class Gorda(BaseModel):

    def __init__(self, mu, X, Nf, mu_FG=None):

        '''
        Sets up the chemical potential for the evaluate() 
        function. pQCD EOS model only possesses the evaluate()
        function since it is already calibrated. 

        Parameters:
        -----------
        mu : numpy.linspace
            The quark chemical potential needed to generate
            the mean and standard deviation of the pressure
            in the pQCD EOS model. 

        Returns:
        --------
        None. 
        '''

        # set the chemical potential (change later if needed)
        self.mu = mu
        self.X = X
        self.Nf = Nf
        if mu_FG is not None:
            self.mu_FG = mu_FG
            self.mU_FG = mu_FG[:,None]

        # instantiate the Gorda class
        self.gorda = PQCD(X=self.X, Nf=self.Nf)
        
        # instantiate number of orders
        self.orders = 3
        
        # set up coefficients and Truncation class
        coeffs = np.array([self.gorda.c_0(self.mu), self.gorda.c_1(self.mu), self.gorda.c_2(self.mu)]).T
        self.trunc = Truncation(x=self.mu, x_FG=self.mu_FG, norders=3, \
                                yref=self.gorda.yref, expQ=self.gorda.expQ, coeffs=coeffs)

        return None
    
    
#     def evaluate(self, input_space=None, N2LO=True, scaled=True):

#         '''
#         The Gorda pQCD model with truncation errors
#         included. Sends back the necessary means and
#         standard deviation values.

#         Coming soon: N3LO.

#         Parameters:
#         -----------
#         input_space : numpy.array
#             The input space. Must be supplied though
#             usually already a class variable from the
#             __init__(). 

#         N2LO : bool
#             If True, returns the mean and std_dev of the N2LO 
#             results only. If otherwise, will return up to and 
#             through N2LO.

#         Returns:
#         --------
#         mean, std_dev : numpy.ndarray
#             The mean and standard deviation values for
#             the pQCD model from Gorda et al. (2021).
#         '''

#         # correct for input_space (for now)
#         if input_space is not None:
#             input_space = None

#         # call the correct interpolation and work through
#         pred, std, underlying_std = self.trunc.gp_interpolation(center=0.0, sd=1.0)
#         mean, coeffs_trunc, std_dev = \
#             self.trunc.uncertainties(expQ=self.gorda.expQ, yref=self.gorda.yref)

#         if N2LO is True:
#             mean = mean[:,2]
#             std_dev = std_dev[:,2]
                
#         if scaled is True:
#             mean = mean/self.gorda.yref(self.mu_FG)
#             std_dev = std_dev/self.gorda.yref(self.mu_FG)
#             return mean, std_dev 
#         else:
#             return mean, std_dev
        
        
    def evaluate(self, input_space=None, N2LO=True, scaled=True):
        
        # trying to write an evaluate function for the 
        # KLW version of the above code
        
        conversion = (1000)**4.0/(197.327)**3.0
        
        # correct input space
        if input_space is not None:
            input_space = None
            
        # call interpolation and work through
        pred, std, underlying_std = self.trunc.gp_interpolation(center=0.0, sd=1.0)
        
        # coeffs and data solved at mu_FG
        coeffs_FG = np.array([self.gorda.c_0(self.mu_FG), self.gorda.c_1(self.mu_FG), self.gorda.c_2(self.mu_FG)]).T
        data_FG = gm.partials(coeffs_FG, ratio=self.gorda.expQ(self.mU_FG), \
                              ref=self.gorda.yref(self.mU_FG), orders=[range(3)])
        _, coeffs_trunc, std_dev = \
            self.trunc.uncertainties(data=data_FG, expQ=self.gorda.expQ, yref=self.gorda.yref)
        
        # fix lower densities 
        for j in range(self.orders):
            for i in range(len(std_dev)):
                if np.isnan(std_dev[i,j]) == True or np.isinf(std_dev[i,j]) == True:
                    std_dev[i,j] = 1e10
        
        # term by term KLW inversion for the pressure
        pressureNLO_term1 = (self.gorda.mu_1(self.mu_FG))* \
                            self.gorda.n_FG_mu(self.mu_FG)/self.gorda.yref(self.mU_FG) # first order #checked
        pressureN2LO_term1 = self.gorda.mu_2(self.mu_FG)* \
                            self.gorda.n_FG_mu(self.mu_FG)/self.gorda.yref(self.mU_FG) # second order #checked
        pressureN2LO_term2 = 0.5*self.gorda.mu_1(self.mu_FG)**2.0 * \
            (self.Nf * 3.0 * self.mu_FG**2.0 / np.pi**2.0)/self.gorda.yref(self.mU_FG) # second order alpha_s
        pressureN2LO_term3 = self.gorda.mu_1(self.mu_FG)* \
                             (self.gorda.c_1(self.mu_FG)*self.gorda.alpha_s(self.mu_FG)\
                             *self.gorda.n_FG_mu(self.mu_FG)/self.gorda.yref(self.mU_FG)) # second order alpha_s #checked

        # organise into powers
        pressure_0 = self.gorda.yref(self.mU_FG) * self.gorda.c_0(self.mu_FG)  #checked
        pressure_1 = self.gorda.yref(self.mU_FG) * (self.gorda.c_0(self.mu_FG) + \
                     self.gorda.c_1(self.mu_FG)*self.gorda.alpha_s(self.mu_FG) \
                     + pressureNLO_term1) #checked
        pressure_2 = self.gorda.yref(self.mU_FG) * (self.gorda.c_0(self.mu_FG) \
                     + self.gorda.c_1(self.mu_FG)*self.gorda.alpha_s(self.mu_FG) \
                        + pressureNLO_term1 + self.gorda.c_2(self.mu_FG) \
                            *self.gorda.alpha_s(self.mu_FG)**2.0 \
                                + pressureN2LO_term1 + pressureN2LO_term2 + pressureN2LO_term3) #checked

        # put these into an array
        mean = np.array([pressure_0, pressure_1, pressure_2]).T
        
        if N2LO is True:
            mean = mean[:,2]
            std_dev = std_dev[:,2]
                
        if scaled is True:
            mean = mean/self.gorda.yref(self.mU_FG)
            std_dev = std_dev/self.gorda.yref(self.mU_FG)
            return mean, std_dev 
        else:
            return mean, std_dev

    # the following functions not used for our models
    def log_likelihood_elementwise(self):
        return None
    def set_prior(self):
        return None