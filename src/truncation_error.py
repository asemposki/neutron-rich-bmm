###########################################################
# Truncation error class to work with the pQCD EOS, but
# also to work more generally with any provided expansion.
# Author : Alexandra Semposki
# Adapted from : gsum tutorial notebooks by J. Melendez,
#                R. J. Furnstahl, D. R. Phillips
# Date : 20 March 2024
# ##########################################################

# imports
import numpy as np
import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# class declaration
class Truncation:
    
    '''
    The truncation error class that wraps the gsum package.
    For use on the pQCD EOS results. 
    
    Parameters:
    -----------
    x : numpy.array
        Quark chemical potential.
        
    x_FG : numpy.2darray
        FG quark chemical potential.
        
    norders : int
        Number of orders used.
        
    orders : list
        The list of orders ([0 1 2], etc.)
        
    yref : function
        Functional form for yref. 
        
    expQ : function
        Functional form for the expansion parameter.
        
    coeffs : numpy.array
        The coefficient array of arrays. 
        Must be transposed when sent in to this
        class.
    
    mask : boolean array
        If using a mask, send the mask in here.
    
    Returns:
    --------
    None.
    '''

    def __init__(self, x, x_FG, norders, orders, yref, expQ, coeffs, mask=None):
        
        # immediately create this 
        self.orders_mask = None
        self.orders = orders
        self.n_orders = norders

        # define class variables for pressure to all orders
        self.x = x
        self.X = x[:,None]
        if x_FG is not None:
            self.x_FG = x_FG
            self.X_FG = x_FG[:,None]
        else:
            self.x_FG = None
            self.X_FG = None
        self.norders = norders

        # declare yref, Q
        self.yref = yref(self.X) 
        self.expQ = expQ(self.X)
        
        ### formalism from gsum docs ###
                
        # separate the coeffs
        self.coeffs_list = []
        
        # if masking, create separate instance of coeffs
        if mask is not None:
            self.coeffs_list_trunc = []
            self.orders_mask = mask
            coeffs_trunc = coeffs[:,self.orders_mask]
            
            for i in range(len(coeffs_trunc.T)):
                self.coeffs_list_trunc.append(coeffs_trunc[:,i])
                
        for i in range(len(coeffs.T)):
            self.coeffs_list.append(coeffs[:,i])

        # construct total arrays of each quantity
        self.coeffs_all = np.array(self.coeffs_list).T
        self.data_all = gm.partials(self.coeffs_all, ratio=self.expQ, ref=self.yref, orders=[range(norders)])
        self.diffs_all = np.array([self.data_all[:, 0], *np.diff(self.data_all, axis=1).T]).T
        
        # save different coeff array for the GP to fit
        if mask is not None:
            self.coeffs_all_trunc = np.array(self.coeffs_list_trunc).T

        # get the "all-orders" curve
        self.data_true = self.data_all[:, -1]

        # specify range
        if mask is not None:
            self.coeffs_trunc = self.coeffs_all_trunc
    
        self.coeffs = self.coeffs_all[:, :norders]
        self.data = self.data_all[:, :norders]
        self.diffs = self.diffs_all[:, :norders]

        return None
    

    def gp_mask(self, mu):

        '''
        The mask array needed to correctly separate our training
        and testing data.

        Parameters:
        -----------
        mu : numpy.ndarray
            The chemical potential linspace needed.

        Returns:
        --------
        self.mask : numpy.ndarray
            The mask for use when interpolating or using the 
            truncated GP.
        '''
        
        # mask for values above 40*n0 only (good)
        low_bound = next(i for i, val in enumerate(mu)
                                  if val > 0.946136639) # value of mean of mu_FG and mu_n at 40*n0
        mu_mask = mu[low_bound:]
#        mu_mask = mu    # no mask applied
        
        # set the mask s.t. it picks the same training point number each time
        mask_num = len(mu_mask) // 1.5 #2.5  # original is 2 here for Nf*alpha_s/pi
        mask_true = np.array([(i) % mask_num == 0 for i in range(len(mu_mask))])  # i-3 for <40n0
        
        # concatenate with a mask over the other elements of mu before low_bound
        mask_false = np.full((1,len(mu[:low_bound])), fill_value=False)
        self.mask = np.concatenate((mask_false[0], mask_true))
#        self.mask = mask_true     #no mask applied
       
        return self.mask 
    
    
    def gp_kernel(self, ls=3.0, sd=0.5, center=0, nugget=1e-10):

        '''
        The kernel that we will use both for interpolating the 
        coefficients and for predicting the truncation error bands.
        This one is unfixed, so the value of the ls obtained here will 
        be used to fix the second run when calling params attribute.

        Parameters:
        -----------
        ls : float
            The lengthscale guess for the kernel.
        
        sd : float
            The scale for the prior.
            
        center : float
            The center value for the prior.
            
        nugget : int, float
            The value of the nugget to send to the 
            Cholesky decomposition.

        Returns:
        --------
        kernel : sklearn object
            The kernel needed for the GPs. 
        '''

        self.ls = ls    # starting guess; can get really close if we set 0.25 and fix it
        self.sd = sd    # makes a difference on the band of the regression curve for c_2 
        self.center = center
        self.nugget = nugget # nugget for the Cholesky, not the kernel 
        kernel = RBF(length_scale=self.ls)# + \
       # WhiteKernel(noise_level=self.nugget, noise_level_bounds='fixed')

        return kernel
    

    def gp_interpolation(self, center=0.0, sd=1.0):

        '''
        The function responsible for fitting the coefficients with a GP
        and predicting at new points. This information will be used in 
        constructing our truncated GP in the function 'Uncertainties'. 

        Parameters:
        -----------
        center : float
            The center value for the prior.
        
        sd : float
            The scale for the prior.

        Returns:
        --------
        pred : numpy.ndarray
            An array of predictions from the GP.

        std : numpy.ndarray
            The standard deviation at the points in 'pred'.

        underlying_std : numpy.ndarray
            The underlying standard deviation of the GP.
        '''

        # interpolate the coefficents using GPs and gsum 
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
        
        # call the mask function to set this for interpolation training
        self.mask = self.gp_mask(self.x)

        # Set up gp objects with fixed mean and standard deviation 
        kernel = self.gp_kernel(ls=3.0, sd=sd, center=0)  
        self.gp_interp = gm.ConjugateGaussianProcess(
            kernel=kernel, center=center, disp=0, df=3.0, scale=sd, nugget=0) 
        
        # fit and predict using the interpolated GP
        if self.orders_mask is not None:
            self.gp_interp.fit(self.X[self.mask], self.coeffs_trunc[self.mask])
        else:
            self.gp_interp.fit(self.X[self.mask], self.coeffs[self.mask])
        pred, std = self.gp_interp.predict(self.X, return_std=True)
        underlying_std = np.sqrt(self.gp_interp.cov_factor_)
        
        # print the kernel parameters for viewing outside the function
        print(self.gp_interp.kernel_)
        print(self.gp_interp.cov_factor_)

        return pred, std, underlying_std
    

    def uncertainties(self, data=None, expQ=None, yref=None, sd=0.5, nugget=1e-10, excluded=None):

        '''
        Calculation of the truncation error bands for the pQCD EOS, using 
        the Gorda et al. (2021) formulation for the pressure.
        This function uses techniques from the gsum package. 

        Parameters:
        -----------
        data : numpy.ndarray
            The data given in an array of arrays for 
            each order-by-order result.
        
        expQ : function
            The functional form of the expansion parameter
            for gsum to use.
        
        yref : function
            The functional form of yref for gsum to use.
         
        sd : float
            The scale for the prior.
        
        nugget : int, float
            The nugget for the Cholesky decomposition.
        
        excluded : list
            The orders we wish to exclude from training
            on in the coefficient arrays. Default is None.
        
        Returns:
        --------
        data : numpy.ndarray
            The data array, containing partials at each order.
        
        self.coeffs : numpy.ndarray
            The values of the coefficents at x.
        
        std_trunc : numpy.ndarray
            The arrays of truncation errors per each order.

        '''
        
        # construct mask
        self.mask = self.gp_mask(self.x)

        # get correct data shape
        if data is None:         
            data = self.data_all[:, :self.n_orders]
        else:
            data = data[:, :self.n_orders]
            
        # save nugget
        self.nugget = nugget

        # set up the truncation GP
        self.kernel = self.gp_kernel(ls=3.0, sd=sd, center=0, nugget=self.nugget) 
        self.trunc_gp = gm.TruncationGP(kernel=self.kernel, ref=yref, \
                            ratio=expQ, disp=0, df=3.0, scale=sd, excluded=excluded, nugget=self.nugget)
        
        self.trunc_gp.fit(self.X[self.mask], data[self.mask], orders=self.orders)
        
        std_trunc = np.zeros([len(self.X), self.n_orders])
        cov_trunc = np.zeros([len(self.X), len(self.X), self.n_orders])
        for i, n in enumerate(self.orders):
            # Only get the uncertainty due to truncation (kind='trunc')
            _, std_trunc[:,n] = self.trunc_gp.predict(self.X, order=n, return_std=True, \
                                                      kind='trunc', pred_noise=True)
            _, cov_trunc[:,:,n] = self.trunc_gp.predict(self.X, order=n, return_std=False, \
                                                        return_cov=True, kind='trunc', pred_noise=True)
            
        # external access without altering return
        self.cov_trunc = cov_trunc
        
        # check the kernel hyperparameters
        print(self.trunc_gp.coeffs_process.kernel_)
        print(self.trunc_gp.coeffs_process.nugget)
        
        return data, self.coeffs, std_trunc, cov_trunc
    
    
    # masking for diagnostics ONLY (taken from Jordan Melendez's gsum code directly)
    def regular_train_test_split(self, x, dx_train, dx_test, offset_train=0, offset_test=0, \
                                 xmin=None, xmax=None):
        train_mask = np.array([(i - offset_train) % dx_train == 0 for i in range(len(x))])
        test_mask = np.array([(i - offset_test) % dx_test == 0 for i in range(len(x))])
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x)
        train_mask = train_mask & (x >= xmin) & (x <= xmax)
        test_mask = test_mask  & (x >= xmin) & (x <= xmax) & (~ train_mask)
        return train_mask, test_mask
    
    
    def diagnostics(self, dx_train=30, dx_test=15):
        
        '''
        The diagnostic function to check the validity of 
        the truncation error obtained via gsum. Uses
        gsum to perform Mahalanobis distance and pivoted
        Cholesky calculations. Plots the results.
        
        Parameters:
        -----------
        dx_train : int
            The number to use as a step size for the
            training data.
        
        dx_test : int
            The number to use as a step size for the
            testing data.
        
        Returns:
        --------
        None
        '''
        
        # set the plot labels
        MD_label = r'$\mathrm{D}_{\mathrm{MD}}^2$'
        PC_label = r'$\mathrm{D}_{\mathrm{PC}}$'

        # set up plotting tools
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['figure.dpi'] = 150   # change for paper plots
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        WIDE_IMG_WIDTH = 800
        NARROW_IMG_WIDTH = 400

        cmaps = [plt.get_cmap(name) for name in ['Oranges', 'Greens', 'Blues', 'Reds']]
        colors = [cmap(0.55 - 0.1 * (i==0)) for i, cmap in enumerate(cmaps)]
        light_colors = [cmap(0.25) for cmap in cmaps]

        edgewidth = 0.6
        text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec='k', lw=0.8)
        
        # call the masking function for diagnostics
        x_train_mask, x_valid_mask = self.regular_train_test_split(self.x, dx_train, dx_test, \
                                                                   offset_train=0, \
                                                                   offset_test=0, xmin=0.88616447)
        
        # print the values for checking (use only ones after 40 n0 again)
        print('Number of points in training set:', np.shape(self.X[self.mask])[0])
        print('Number of points in validation set:', np.shape(self.X[x_valid_mask])[0])

        print('\nTraining set: \n', self.X[self.mask])
        print('\nValidation set: \n', self.X[x_valid_mask])

        # overwrite training mask with the original mask for keeping range correct
        x_train_mask = self.mask 

        # check if the two arrays have equal elements
        for i in self.X[x_train_mask]:
            for j in self.X[x_valid_mask]:
                if i == j:
                    print('Found an equal value!')
    
        # already fit the GP kernel, do diagnostics directly now 
        underlying_std = np.sqrt(self.trunc_gp.coeffs_process.cov_factor_)
        print(underlying_std)
        print(self.trunc_gp.coeffs_process.nugget)

        gp_diagnostic = self.trunc_gp.coeffs_process  # set equal here (same object)
        print('Calculated value :', gp_diagnostic.df_ * gp_diagnostic.scale_**2 / (gp_diagnostic.df_ + 2))

        # Print out the kernel of the fitted GP
        print('Trained kernel: ', self.trunc_gp.coeffs_process.kernel_)
        print('Scale: ', gp_diagnostic.scale_)
        
        # MD diagnostic plotting
        mean_underlying = gp_diagnostic.mean(self.X[x_valid_mask])
        cov_underlying = gp_diagnostic.cov(self.X[x_valid_mask])
        print('Condition number:', np.linalg.cond(cov_underlying))
        
        # coeffs coming from initial set up
        gdgn = gm.GraphicalDiagnostic(self.coeffs[x_valid_mask], mean_underlying, cov_underlying, \
                                      colors=colors,
                                      gray='gray', black='k')

        def offset_xlabel(ax):
            ax.set_xticks([0])
            ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
            ax.tick_params(axis='x', length=0)
            return ax

        fig, ax = plt.subplots(figsize=(1, 3.2))
        ax = gdgn.md_squared(type='box', trim=False, title=None, xlabel=MD_label)
        offset_xlabel(ax)
        ax.set_ylim(0, 20)
        
        # Pivoted Cholesky as well
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            gdgn.pivoted_cholesky_errors(ax=ax, title=None)
            ax.text(0.04, 0.967, PC_label, bbox=text_bbox, transform=ax.transAxes, va='top', ha='left')
            plt.show()
        
        return None
