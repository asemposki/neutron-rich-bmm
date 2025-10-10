from operator import itemgetter
from scipy.linalg import cho_solve, cholesky, solve_triangular
import numpy as np
from scipy import stats
import scipy as scipy
import warnings
import numdifftools as ndt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.base import clone
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state

class Priors:

    def __init__(self, prior_name):

        # take in lengthscale only for this prior
        ls = np.exp(theta[1])
        a = np.exp(self.kernel_.bounds[1,0])
        b = np.exp(self.kernel_.bounds[1,1])

        if prior_name == 'matern52_norm15':
            if self.cutoff == 20:
                #return self.luniform_ls(ls, a, b) + stats.norm.logpdf(ls, 1.5, 0.38) # 20n0 0.8
                return self.luniform_ls(ls, a, b) + stats.norm.logpdf(ls, 0.65, 0.15)#0.65, 0.15)#1.26, 0.15) # 20n0 0.8

            elif self.cutoff == 40:
                #return self.luniform_ls(ls, a, b) + stats.norm.logpdf(ls, 2.4, 0.38)  # 40n0  0.95, 0.1
                return self.luniform_ls(ls, a, b) + stats.norm.logpdf(ls, 1.03, 0.15) #1.03, 0.15)#1.65, 0.15)  # 40n0  0.95, 0.1