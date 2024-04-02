# ## pQCD model class for Taweret
# ## Written by : Alexandra Semposki
# ## Last edited : 16 January 2024

# imports and paths
import numpy as np
from pqcd import Kurkela
import sys

from Taweret.Taweret.core.base_model import BaseModel


# begin model setup
class PQCD_model(BaseModel):

    def __init__(self, mu, X, Nf):

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
            The renormalization scale parameter.
        
        Nf : int
            The quark flavour. 

        Returns:
        --------
        None. 
        '''

        # set the chemical potential (change later if needed)
        self.mu = mu
        self.X = X
        self.Nf = Nf

        # instantiate the Kurkela class
        self.kurkela = Kurkela(X=self.X, Nf=self.Nf)

        return None


    def evaluate(self, input_space=None, wrt_dens=False, N2LO=True, scaled=True):

        '''
        Returns the mean and standard deviation of the pQCD EOS
        in terms of pressure wrt quark chemical potential.
        In this EOS, quarks are massless. 

        Mean: calculated using the Kurkela formalism. 

        Standard deviation: calculated via the truncation error
                            models in the gsum package.

        Parameters:
        -----------
        input_space : numpy.array
            The input space for this model. Not needed for this function,
            but necessary for Taweret.
            
        wrt_dens : bool
            Option to solve Kurkela pQCD EOS wrt the number density
            instead of the chemical potential. Involves solving wrt
            chemical potential and inverting the result for density.

        N2LO : bool
            If True, returns the mean and std_dev of the N2LO results only.
            If otherwise, will return up to and through N2LO.
            
        scaled : bool
            If the data is scaled or not. Default is True.

        Returns:
        --------
        mean, std_dev : numpy.ndarray
            The mean and standard deviation of the pressure. 
        '''
        
        # correct for input_space (for now)
        if input_space is not None:
            input_space = None

        # mean and standard deviation if wrt density
        if wrt_dens is False:
            if scaled is True:
                mean, _, std_dev = self.kurkela.uncertainties(mu=self.mu, n_orders=3)

                if N2LO is True:
                    mean = mean[:,2]
                    std_dev = std_dev[:,2]
                    
                    if scaled is True:
                        mean = mean/self.kurkela.yref(self.kurkela.mu_arr)
                        std_dev = std_dev/self.kurkela.yref(self.kurkela.mu_arr)
                    return mean, std_dev 
                
                else:
                    return mean, std_dev
        
        # mean and standard deviation if wrt mu
        else:
            
            # convert from chemical potential to density
            ### TO DO for uncertainty part
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
