# import external dependences
import math
import warnings
import numpy as np
import numdifftools as ndt
from scipy.spatial.distance import cdist, pdist, squareform

from sklearn.utils import check_random_state
from sklearn.gaussian_process.kernels import Kernel, RBF
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
    
    # for now can input the choices, but later will need optimizing (### get theta array figured out ###)
    def __init__(self, ls1, ls2, cbar1, cbar2, width, changepoint):
        self.ls1 = ls1
        self.ls2 = ls2
        self.cbar1 = cbar1
        self.cbar2 = cbar2
        self.width = width
        self.changepoint = changepoint 
        
        self.type = 'sigmoid'
        
        return None
    
    # call the function, see what happens
    def __call__(self, X, Y=None, eval_gradient=False):  # eval_gradient = True needed for parameter optimization
        
        # check the dimensions
        X = np.atleast_2d(X)
        
        # this should work for all kernels (not just stationary)
        if Y is None:
            Y = X
            
        # initialize the K kernel matrix (len(tr_data), len(tr_data))
        K = np.zeros([len(X), len(Y)])
        
        # assign the stationary kernels (chiral and pQCD)
        self.K1 = (C(constant_value=self.cbar1, constant_value_bounds='fixed') \
                  * RBF(length_scale=self.ls1, length_scale_bounds='fixed'))(X,Y)
#        self.K1 = RBF(length_scale=self.ls1, length_scale_bounds='fixed')(X,Y)
        
        self.K2 = (C(constant_value=self.cbar2, constant_value_bounds='fixed') \
                    * RBF(length_scale=self.ls2, length_scale_bounds='fixed'))(X,Y)
#        self.K2 = RBF(length_scale=self.ls2, length_scale_bounds='fixed')(X,Y)
        
        self.K3 = C(constant_value=0.0, constant_value_bounds='fixed')(X,Y)
        
        # if statement for cases
        if self.type == 'theta':
         
            # assign Heaviside functions like sigmoid below (careful with assigning this!)
            K = np.outer(np.ones(len(X)) - np.heaviside(X - self.changepoint, 1).T, \
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
            
            # loop for each point
            K = np.outer((np.ones(len(X)) - sigmoid(X, self.changepoint, self.width).T), \
                              (np.ones(len(Y)) - sigmoid(Y, self.changepoint, self.width).T)) * self.K1 + np.outer(sigmoid(X, \
                             self.changepoint, self.width).T, sigmoid(Y, self.changepoint, self.width).T) * self.K2
        
            #print('From the kernel code: ', K)
            #print('Diagonals from K: ', np.diag(K))
                                        
        # only for when optimization of hyperparameters is needed
        if eval_gradient:
            
            ### write the gradient wrt changepoint here! ###

#             grad_ls1 = -self.K1 * (np.square(X[:, np.newaxis] - X[np.newaxis, :]) / (2 * self.ls1**3))
#             grad_ls2 = -self.K2 * (np.square(X[:, np.newaxis] - X[np.newaxis, :]) / (2 * self.ls2**3))

#             # calculate with changepoints involved here
#             grad_ls1_cp = np.where(X[:, 0] < self.changepoint, grad_ls1, 0)
#             grad_ls2_cp = np.where(X[:, 0] >= self.changepoint, grad_ls2, 0)
            
#             # store the gradient
#             grad_total = np.dstack((grad_ls1_cp, grad_ls2_cp))

            return K, grad_total
        
        return K

    # diagonal function
    def diag(self, X):
        return np.ones(X.shape[0])  # clearly einsum doesn't work (hmm...)

    # stationary function needed for base class
    def is_stationary(self):
        return False