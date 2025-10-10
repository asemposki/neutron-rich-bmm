###########################################################
# Gorda pQCD model for Taweret implementation
# Author : Alexandra Semposki
# Date : 16 January 2024
###########################################################

# Note: this class is not used currently to generate
# the results in the paper, but is left here in case one 
# wishes to use it. 

# imports
import numpy as np
import gsum as gm
import sys
from pqcd_reworked import PQCD
from truncation_error import Truncation

sys.path.append('../../Taweret')
from Taweret.core.base_model import BaseModel

# wrapper class for pQCD EOS
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
            
        X : int
            The value of the renormalization scale parameter.
        
        Nf : int
            The number of flavours of quarks considered.
            
        mu_FG : numpy.ndarray
            The FG chemical potential array.

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
    
    
    def evaluate(self, input_space=None, N2LO=True, scaled=True):
        
        '''
        The evaluation function for Taweret to obtain the 
        calibrated mean and variance of the pQCD model.
        
        Parameters:
        -----------
        input_space: numpy.ndarray
            The number density array.
        
        N2LO : bool
            If we only want the data from the pQCD pressure 
            at N2LO. Default is True.
            
        scaled : bool
            If the data is scaled, this is True. Else, it is False.
            Default is True.
        
        Returns:
        --------
        mean, std_dev: numpy.ndarray
            The mean and standard deviations at the selected
            points in the input space of the pQCD pressure.
        '''
        
        # KLW formalism for the pQCD EOS pressure
        conversion = (1000)**4.0/(197.327)**3.0
        
        # correct input space
        if input_space is not None:
            input_space = None
            
        # call interpolation and work through
        _, _, _ = self.trunc.gp_interpolation(center=0.0, sd=1.0)
        
        # coeffs and data solved at mu_FG
        coeffs_FG = np.array([self.gorda.c_0(self.mu_FG), self.gorda.c_1(self.mu_FG), self.gorda.c_2(self.mu_FG)]).T
        data_FG = gm.partials(coeffs_FG, ratio=self.gorda.expQ(self.mU_FG), \
                              ref=self.gorda.yref(self.mU_FG), orders=[range(3)])
        _, coeffs_trunc, std_dev = \
            self.trunc.uncertainties(data=data_FG, expQ=self.gorda.expQ, yref=self.gorda.yref)
        
        # fix lower densities when run below n = 0.16 fm^-3
        for j in range(self.orders):
            for i in range(len(std_dev)):
                if np.isnan(std_dev[i,j]) == True or np.isinf(std_dev[i,j]) == True:
                    std_dev[i,j] = 1e10
        
        # KLW pressure call
        pressure_n = self.gorda.pressure_KLW(self.mu_FG)
        
        # put these into an array
        mean = np.array([pressure_n["LO"], pressure_n["NLO"], pressure_n["N2LO"]]).T
        
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
        '''
        The log likelihood function that would calculate
        this quantity in Taweret.
        '''
        return None
    
    def set_prior(self):
        '''
        The prior function to set a prior for 
        Taweret to take in for this model.
        '''
        return None