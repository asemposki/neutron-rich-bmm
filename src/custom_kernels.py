# import external dependences
import math
import warnings
import numpy as np
import numdifftools as ndt
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.utils import check_random_state
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter
from sklearn.gaussian_process.kernels import ConstantKernel as C

# set up a wrapper for sklearn kernel class
class Changepoint(Kernel): 
    
    r'''
    Designs a non-stationary changepoint kernel that inherits 
    from the sklearn RBF Kernel class.
    
    The kernel is given by:
    .. math::
        k(x_i, x_j) = (1 - \\sigma(x_i)) * K1(x_i,x_j) * 
                      (1 - \\sigma(x_j)) + \\sigma(x_i) * K2(x_i,x_j) 
                      * \\sigma(x_j)

    where K1 is the first kernel and K2 is the second kernel, with the
    changepoint defined by the sigmoid function.
    '''
    
    # import hyperparameter names here
    def __init__(self, ls1, ls2, cbar1, cbar2, width=1.0, changepoint=1.0,
                changepoint_bounds=(1e-5, 1e5), width_bounds=(1e-5, 1e5)):
        
        # will not touch these for now
        self.ls1 = ls1
        self.ls2 = ls2
        self.cbar1 = cbar1
        self.cbar2 = cbar2
        
        # optimizable parameters
        self.width = width
        self.width_bounds = width_bounds
        self.changepoint = changepoint 
        self.changepoint_bounds = changepoint_bounds
        
        # which kernel type am I using (make into 2 classes later)
        self.type = 'sigmoid'
        
        return None
    
    @property
    def anisotropic(self):
        return np.iterable(self.changepoint) and len(self.changepoint) > 1
    
    # get hyperparmeters to be optimized, leave option for fixed values
    @property
    def hyperparameter_changepoint(self):
        if self.anisotropic:
            return Hyperparameter(
                "changepoint",
                "numeric",
                self.changepoint_bounds,
                len(self.changepoint),
            )
        return Hyperparameter("changepoint", "numeric", self.changepoint_bounds)
    
    @property
    def hyperparameter_width(self):
        if self.anisotropic:
            return Hyperparameter(
                "width",
                "numeric",
                self.width_bounds,
                len(self.width),
            )
        return Hyperparameter("width", "numeric", self.width_bounds)
    
    # call the function, see what happens
    def __call__(self, X, Y=None, eval_gradient=False):  
        
        # check the dimensions
        X = np.atleast_2d(X)
        
        # this should work for all kernels
        if Y is None:
            Y = X
            
        # initialize the K kernel matrix (len(tr_data), len(tr_data))
        self.K = np.zeros([len(X), len(Y)])
        
        # assign the stationary kernels (chiral and pQCD)
        self.K1 = (C(constant_value=self.cbar1, constant_value_bounds='fixed') \
                  * RBF(length_scale=self.ls1, length_scale_bounds='fixed'))(X,Y)
        
        self.K2 = (C(constant_value=self.cbar2, constant_value_bounds='fixed') \
                    * RBF(length_scale=self.ls2, length_scale_bounds='fixed'))(X,Y)
        
        self.K3 = C(constant_value=0.0, constant_value_bounds='fixed')(X,Y)
        
        # if statement for cases
        if self.type == 'theta':
         
            # assign Heaviside functions like sigmoid below (careful with assigning this!)
            self.K = np.outer(np.ones(len(X)) - np.heaviside(X - self.changepoint, 1).T, \
                         np.ones(len(Y)) - np.heaviside(Y - self.changepoint, 1).T) * \
                         self.K1 + np.outer(np.heaviside(X - self.changepoint, 1).T, \
                         np.heaviside(Y - self.changepoint, 1).T) * self.K2 + \
                         np.outer(np.ones(len(X)) - np.heaviside(X - self.changepoint, 1).T, \
                         np.heaviside(Y - self.changepoint, 1).T) \
                         * self.K3 + np.outer(np.heaviside(X - self.changepoint, 1).T, \
                         np.ones(len(Y)) - np.heaviside(Y - self.changepoint, 1).T) * self.K3

        elif self.type == 'sigmoid':

            # sigmoid bilinear function
            def sigmoid(x0, k, dens):
                return 1.0 / (1.0 + np.exp(-(dens-x0)/k))
            
            # sigmoid derivative function
            def sig_grad(dens, x0, k):
                return - (1.0 / k) * (1.0 + np.exp(-(dens-x0)/k))**(-2.0)
            
            # loop for each point
            self.K = np.outer((np.ones(len(X)) - sigmoid(self.changepoint, self.width, X).T), \
                         (np.ones(len(Y)) - sigmoid(self.changepoint, self.width, Y).T)) * self.K1 + 
                          np.outer(sigmoid(self.changepoint,self.width, X).T, \                                                   
                          sigmoid(self.changepoint, self.width, Y).T) * self.K2
                                        
        # only for when optimization of hyperparameters is needed
        if eval_gradient:  # for the single hyperparameter now that is varying (changepoint)
            
            if self.hyperparameter_changepoint.fixed:
                # hyperparameter changepoint kept fixed (no gradient)
                return self.K, np.empty((X.shape[0], X.shape[0], 0)) # not sure of this
                
            elif not self.anisotropic or length_scale.shape[0] == 1:
                if self.type == 'sigmoid':
                    def K_sig_grad_cp(cp, *args):
                        sig_grad = ndt.Derivative(sigmoid, )
                        return None
#                     K_gradient = (np.outer((-sig_grad(X, self.changepoint, self.width).T), \
#                                   (np.ones(len(Y)) - sigmoid(Y, self.changepoint, self.width).T)) 
#                                   * self.K1 + np.outer((np.ones(len(X)) - sigmoid(X, self.changepoint, self.width).T), \
#                                   (-sig_grad(Y, self.changepoint, self.width).T)) * self.K1 
#                                   + np.outer(sig_grad(X, self.changepoint, self.width).T, 
#                                    sigmoid(Y, self.changepoint, self.width).T) * self.K2 + 
#                                   np.outer(sigmoid(X, self.changepoint, self.width).T, 
#                                    sig_grad(Y, self.changepoint, self.width).T) * self.K2)[:,:,np.newaxis]
                    return self.K, K_gradient
                elif self.type == 'theta':
                    K_gradient = None
                    raise ValueError('''The gradient cannot be evaluated for the Heaviside changepoint kernel.'''
                                      ''' Optimization cannot be performed using this kernel choice.''')
                
            elif self.anisotropic:
                raise ValueError('The kernel has not been implemented for anisotropic cases.')
                    
        else:
            return self.K  # no gradient returned
        
    # diagonal function (general for the nonstationary case)
    def diag(self, X):
        return np.apply_along_axis(self, 1, X).ravel() # makes sense---> X = Y
        
    # stationary function needed for base class
    def is_stationary(self):
        return False
    
    # this returns the parameter value when trained
    def __repr__(self):
        if self.anisotropic:
            return "{0}(changepoint=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.changepoint)),
            )
        else:  # isotropic
            return "{0}(changepoint={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.changepoint)[0]
            )