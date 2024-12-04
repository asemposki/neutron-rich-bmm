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
    def __init__(self, ls1, ls2, cbar1, cbar2, changepoint=1.0, changepoint_bounds=(1e-5, 1e5), 
                 width=1.0, width_bounds=(1e-5, 1e5)):
        
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
            def sigmoid(dens, x0, k):
                return 1.0 / (1.0 + np.exp(-(dens-x0)/k))
            
            # sigmoid deriv for cp and width
            def sig_grad(dens, x0, k, deriv='cp'):
                if deriv == 'cp':
                    return -(np.exp(-(dens-x0)/k)/k) * (1.0 + np.exp(-(dens-x0)/k))**(-2.0)
                elif deriv == 'w':
                    return -(np.exp(-(dens-x0)/k)) * (dens-x0)/k**2.0 * (1.0 + np.exp(-(dens-x0)/k))**(-2.0)
            
            # define kernel matrix
            self.K = np.outer((np.ones(len(X)) - sigmoid(X, self.changepoint, self.width).T), \
                              (np.ones(len(Y)) - sigmoid(Y, self.changepoint, self.width).T)) \
            * self.K1 + np.outer(sigmoid(X, self.changepoint,self.width).T, \
                                 sigmoid(Y, self.changepoint, self.width).T) * self.K2
                                        
        # only for when optimization of hyperparameters is needed (DON'T TOUCH THIS AGAIN YOU IDIOT)
        if eval_gradient:

            # check if this is correct or if it should be log of params somehow encoded (YES)
            if self.anisotropic is False:

                # go into the function type
                if self.type == 'sigmoid':
                    
                    # gradient wrt changepoint
                    if self.hyperparameter_changepoint.fixed is False:
                        K_gradient_cp = (self.changepoint*(np.outer((-sig_grad(X, self.changepoint, self.width).T), \
                                            (np.ones(len(Y)) - sigmoid(Y, self.changepoint, self.width).T)) \
                                    * self.K1 + np.outer((np.ones(len(X)) - \
                                                            sigmoid(X, self.changepoint, self.width).T), \
                                                        -sig_grad(Y, self.changepoint, self.width).T) \
                                    * self.K1 + np.outer(sig_grad(X, self.changepoint,self.width).T,  \
                                                        sigmoid(self.changepoint, self.width, Y).T) \
                                    * self.K2 + np.outer(sigmoid(X, self.changepoint,self.width).T, \
                                                        sig_grad(Y, self.changepoint, self.width).T) \
                                    * self.K2))[:,:,np.newaxis]
                    else:
                        K_gradient_cp = np.empty((self.K.shape[0], self.K.shape[1], 0))  # no gradient
                    
                    # gradient wrt width
                    if self.hyperparameter_width.fixed is False:
                        self.hey = 1.0
                        self.K_gradient_w = (self.width*(np.outer((-sig_grad(X, self.changepoint, self.width, 'w').T), \
                                            (np.ones(len(Y)) - sigmoid(Y, self.changepoint, self.width).T)) \
                                    * self.K1 + np.outer((np.ones(len(X)) - \
                                                            sigmoid(X, self.changepoint, self.width).T), \
                                                        -sig_grad(Y, self.changepoint, self.width, 'w').T) \
                                    * self.K1 + np.outer(sig_grad(X, self.changepoint, self.width, 'w').T,  \
                                                        sigmoid(self.changepoint, self.width, Y).T) \
                                    * self.K2 + np.outer(sigmoid(X, self.changepoint,self.width).T, \
                                                        sig_grad(Y, self.changepoint, self.width, 'w').T) \
                                    * self.K2))[:,:,np.newaxis]
                    else:
                        self.K_gradient_w = np.empty((self.K.shape[0], self.K.shape[1], 0))   # no gradient

                    # full gradient return
                    return self.K, np.dstack((K_gradient_cp, self.K_gradient_w))  # is this ordered correctly? how do we know? check?                    
                
                elif self.type == 'theta':
                    raise ValueError('''The gradient cannot be evaluated for the Heaviside changepoint kernel.'''
                                    ''' Optimization cannot be performed using this kernel choice.''')
            
            elif self.anisotropic:
                raise ValueError('The kernel has not been implemented for anisotropic cases.')

        # if eval_gradient is false, return no gradient            
        else:
            return self.K
        
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
            return "{0}(changepoint={1:.3g}, width={2:.3g})".format(
                self.__class__.__name__, self.changepoint, self.width
            )
