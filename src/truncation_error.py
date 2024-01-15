###########################################################
# Truncation error class to work with the pQCD EOS, but
# also to work more generally with any provided expansion.
# Author : Alexandra Semposki
# Adapted from : gsum tutorial notebooks by J. Melendez,
#                R. J. Furnstahl, D. R. Phillips
# Date : 01 August 2023
###########################################################

# imports
import numpy as np
import gsum as gm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# class declaration
class Truncation:

    def __init__(self, x, x_FG, norders, yref, expQ, coeffs):

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
        
        # separate the coeffs
        self.coeffs_list = []

        for i in range(norders):
            self.coeffs_list.append(coeffs[:,i])

        # construct arrays of each quantity
        self.coeffs_all = np.array(self.coeffs_list).T
        self.data_all = gm.partials(self.coeffs_all, ratio=self.expQ, ref=self.yref, orders=[range(norders)])
        self.diffs_all = np.array([self.data_all[:, 0], *np.diff(self.data_all, axis=1).T]).T

        # Get the "all-orders" curve
        self.data_true = self.data_all[:, -1]

        # specify range
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
        print(mu)
        low_bound = next(i for i, val in enumerate(mu)
                                  if val > 0.88616447)
        mu_mask = mu[low_bound:]
        
        # set the mask s.t. it picks the same training point number each time
        mask_num = len(mu_mask) // 2  # original is 2 here
        mask_true = np.array([(i) % mask_num == 0 for i in range(len(mu_mask))])
        
        # concatenate with a mask over the other elements of mu before low_bound
        mask_false = np.full((1,len(mu[:low_bound])), fill_value=False)
        self.mask = np.concatenate((mask_false[0], mask_true))
       
        return self.mask 
    
    
    def gp_kernel_primo(self, ls=3.0, sd=0.2, center=0, nugget=1e-6):

        '''
        The kernel that we will use both for interpolating the 
        coefficients and for predicting the truncation error bands.
        This one is unfixed, so the value of the ls obtained here will 
        be used to fix the second run when calling params attribute.

        Parameters:
        -----------
        None.

        Returns:
        --------
        self.kernel : sklearn object
            The kernel needed for the GPs in both 'uncertainties' and 
            'gp_interpolation'. 
        '''

        self.ls = ls    # starting guess; can get really close if we set 0.25 and fix it
        self.sd = sd    # makes a difference on the band of the regression curve for c_2 
        self.center = center
        self.nugget = nugget #nugget  # nugget goes to Cholesky decomp, not the kernel (kernel has own nugget)
        self.kernel = RBF(length_scale=self.ls) + \
        WhiteKernel(noise_level=self.nugget) # letting this vary 

        return self.kernel
    

    def gp_interpolation(self, center=0.0, sd=1.0):

        '''
        The function responsible for fitting the coefficients with a GP
        and predicting at new points. This information will be used in 
        constructing our truncated GP in the function 'Uncertainties'. 

        Parameters:
        -----------
        x : numpy.ndarray
            The input linspace needed.
            
        kernel : obj
            The kernel needed for the interpolation GP. Can be fed in 
            from the outside for specific parameter alterations.

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
        self.kernel = self.gp_kernel_primo(ls=1.0, sd=0.2, center=0., nugget=1e-4)  
        self.gp_interp = gm.ConjugateGaussianProcess(
            kernel=self.kernel, center=center, disp=0, df=3.0, scale=sd, nugget=0) 
        
        # fit and predict using the interpolated GP
        self.gp_interp.fit(self.X[self.mask], self.coeffs[self.mask])
        pred, std = self.gp_interp.predict(self.X, return_std=True)
        underlying_std = np.sqrt(self.gp_interp.cov_factor_)
        
        # print the kernel parameters for viewing outside the function
        print(self.gp_interp.kernel_)
        print(self.gp_interp.cov_factor_)

        # extract the kernel for later use 
        self.kernel = self.gp_interp.kernel_

        return pred, std, underlying_std
    

    def uncertainties(self, data, expQ, yref):

        '''
        Calculation of the truncation error bands for the pQCD EOS, using 
        the Gorda et al. (2021) formulation for the pressure.
        This function uses techniques from the gsum package. 

        Parameters:
        -----------
        x : numpy.ndarray
            The linspace of input variable needed.
        
        n_orders : int
            The highest order to which the pressure EOS is calculated.
            
        kernel : obj
            The kernel needed for the interpolation and truncation GP.
            Can be fed in from the outside to change parameters.
        
        Returns:
        --------
        data : numpy.ndarray
            The data array, containing partials at each order.
        
        self.coeffs : numpy.ndarray
            The values of the coefficents at x.
        
        std_trunc : numpy.ndarray
            The arrays of truncation errors per each order.

        '''

        # orders
        n_orders = self.norders
        orders = np.arange(0, n_orders)
        self.orders = orders

        # construct the mask
        if self.x_FG is None:
            self.x_FG = self.x
            self.X_FG = self.X
        self.mask = self.gp_mask(self.x_FG)

        # get correct data shape
        if data is None:
            data = self.data_all[:, :n_orders]
        else:
            data = data[:, :n_orders]

        # set up the truncation GP (from interpolation one so using the same kernel as for all orders in P)
        trunc_gp = gm.TruncationGP(kernel=self.kernel, ref=yref, \
                            ratio=expQ, disp=0, df=3, scale=1, optimizer=None)
        trunc_gp.fit(self.X_FG[self.mask], data[self.mask], orders=orders)  

        std_trunc = np.zeros([len(self.X_FG), n_orders])
        cov_trunc = np.zeros([len(self.X_FG), len(self.X_FG), n_orders])
        for i, n in enumerate(orders):
            # Only get the uncertainty due to truncation (kind='trunc')
            _, std_trunc[:,n] = trunc_gp.predict(self.X_FG, order=n, return_std=True, kind='trunc')
            _, cov_trunc[:,:,n] = trunc_gp.predict(self.X_FG, order=n, return_std=False, return_cov=True, kind='trunc')
            
        # external access without altering return
        self.cov_trunc = cov_trunc
        
        return data, self.coeffs, std_trunc
    
    
    # masking for diagnostics ONLY (taken from Jordan Melendez's gsum code directly)
    def regular_train_test_split(self, x, dx_train, dx_test, offset_train=0, offset_test=0, xmin=None, xmax=None):
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
        x_train_mask, x_valid_mask = self.regular_train_test_split(self.x_FG, dx_train, dx_test, offset_train=1, offset_test=1)
        
        # print the values for checking
        print('Number of points in training set:', np.shape(self.X_FG[self.mask])[0])
        print('Number of points in validation set:', np.shape(self.X_FG[x_valid_mask])[0])

        print('\nTraining set: \n', self.X_FG[self.mask])
        print('\nValidation set: \n', self.X_FG[x_valid_mask])

        # overwrite the training mask with the original mask (will yield 6 points)
        x_train_mask = self.mask 

        # check if the two arrays have equal elements
        for i in self.X_FG[x_train_mask]:
            for j in self.X_FG[x_valid_mask]:
                if i == j:
                    print('Found an equal value!')

        # call same kernel for diagnostics (should already be set...remove if true...)
        self.gp_interp.fit(self.X_FG[x_train_mask], self.coeffs[x_train_mask])
        pred, std = self.gp_interp.predict(self.X_FG, return_std=True)
        underlying_std = np.sqrt(self.gp_interp.cov_factor_)
        print(np.sqrt(self.gp_interp.cov_factor_))

        # plot the result of the coefficient interpolation
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.set_xlim(0.4, max(self.X_FG))
        ax.set_xlabel(r'$\mu(n)$')
        for i, n in enumerate(self.orders):
            ax.plot(self.x_FG, pred[:, i], c=colors[i], zorder=i-5, ls='--')
            ax.plot(self.x_FG, self.coeffs[:, i], c=colors[i], zorder=i-5)
            ax.plot(self.x_FG[x_train_mask], self.coeffs[x_train_mask, i], c=colors[i], zorder=i-5, ls='', marker='o',
                    label=r'$c_{}$'.format(n))
            ax.fill_between(self.x_FG, pred[:, i] + 1.96*std, pred[:, i] - 1.96*std, zorder=i-5,
                             facecolor=light_colors[i], edgecolor=colors[i], lw=edgewidth, alpha=1)

        # Format
        ax.axhline(2*underlying_std, 0, 1, c='gray', zorder=-10, lw=1)
        ax.axhline(-2*underlying_std, 0, 1, c='gray', zorder=-10, lw=1)
        ax.axhline(0, 0, 1, c='k', zorder=-10, lw=1)
        ax.legend(ncol=2, borderaxespad=0.5, borderpad=0.4)
        fig.tight_layout();
        
        print(r'Std. dev. expected value:', underlying_std)
        print('Calculated value :', np.sqrt(self.gp_interp.df_ * self.gp_interp.scale_**2 / (self.gp_interp.df_ + 2)))

        # Print out the kernel of the fitted GP
        self.gp_interp.kernel_
        
        # MD diagnostic plotting
        mean_underlying = self.gp_interp.mean(self.X_FG[x_valid_mask])
        cov_underlying = self.gp_interp.cov(self.X_FG[x_valid_mask])
        print(cov_underlying)
        print('Condition number:', np.linalg.cond(cov_underlying))

        gdgn = gm.GraphicalDiagnostic(self.coeffs[x_valid_mask], mean_underlying, cov_underlying, colors=colors,
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
        fig.tight_layout();
        
        # Pivoted Cholesky as well
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            gdgn.pivoted_cholesky_errors(ax=ax, title=None)
            ax.text(0.04, 0.967, PC_label, bbox=text_bbox, transform=ax.transAxes, va='top', ha='left')
            fig.tight_layout();
            plt.show()
        
        return None