# classes to solve the pQCD EOS wrt mu
# Written by: Alexandra Semposki
# Date: 27 February 2023
# Last updated: 09 June 2023 

# !!! Note that mu = mu_q in this code
# so we must convert outside of these functions
# if desiring to use mu_B. 

# import necessary packages
import numpy as np
import matplotlib as mpl
import scipy.integrate as sc
import gsum as gm
import scipy.optimize as opt
from collections import defaultdict
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# matplotlib settings (remove if not working)
mpl.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['legend.fancybox'] = True

# build class
class PQCD:

    def __init__(self, X=2, Nf=3):

        '''
        Initialize the class so that mu and lambda_bar
        have been set and constants have been defined for
        the rest of the code.

        :Example:
            PQCD(mu=np.linspace(), X=1)

        Parameters:
        -----------
        mu : numpy.linspace
            Range of values to use for mu. Units: GeV. 
        
        X : int
            The value of the coefficient multiplying mu
            to determine the renormalization scale. Default is 2,
            as in Kurkela et al. (2010).

        Nf : int
            The number of different quark flavours being 
            considered. Default is 3. 

        Returns:
        --------
        None.
        '''

        # initialize constants and lambda_bar
        self.X = X
        self.Nc = 3
        self.Nf = Nf
        self.Ca = self.Nc
        self.Cf = (self.Nc**2.0 - 1.0)/(2.0*self.Nc)
        self.lambda_MS = 0.38 #[GeV]

        self.beta0 = (11.0*self.Nc - 2.0*self.Nf)/3.0
        
        return None

    # define a function for alpha_s
    def alpha_s(self, mu, loop=2):

        '''
        The function for alpha_s with an option for either 
        first order or second order (loops) inclusion.

        :Example:
            PQCD.alpha_s(mu, loop=1)

        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential. 
            
        loop : int
            The order at which alpha_s is calculated. Default is 2.
            
        Returns:
        --------
        alpha_s : numpy.ndarray
            The values of alpha_s for the input
            chemical potential.
        '''

        lambda_bar = self.X * mu

        if (loop == 1):
            beta1 = 0.0
        elif (loop == 2):
            beta1 = (17.0/3.0)*self.Ca**2.0 - self.Cf*self.Nf - (5.0/3.0)*(self.Ca*self.Nf)
        else:
            raise ValueError('Loop selected must be 1 or 2.')

        # renormalization scale
        ell = np.log((lambda_bar**2.0)/self.lambda_MS**2.0)

        # running coupling constant
        alpha_s = ((4.0*np.pi)/(self.beta0*ell)) * (1.0 - (2.0*beta1*np.log(ell))/(ell*self.beta0**2.0))      

        return alpha_s


    # define the three number density terms from Kurkela et al. (2010)
    def n_FG(self, mu):

        '''
        The zeroth order (LO) term in the number density, from Eq. (59)
        in Kurkela et al. (2010). Aka the FG contribution.

        :Example:
            PQCD.n_FG(mu=np.linspace())

        Parameters:
        -----------
        mu : numpy.linspace
            The original mu range. 

        Returns:
        --------
        n_FG : numpy.ndarray
            The FG contribution to the number density.
        '''

        n_FG = (3.0 * mu**3.0) / np.pi**2.0

        return n_FG


    def n_1(self, mu):

        '''
        The first order (NLO) term in the number density of Eq. (59) in
        Kurkela et al. (2010).

        :Example:
            PQCD.n_1(self, mu=np.linspace())

        Parameters:
        -----------
        mu : numpy.linspace
            The original mu range.

        Returns:
        --------
        n_1 : numpy.ndarray
            The first order contribution to the number density.
        '''

        n_1 = self.n_FG(mu) * (1.0 - (2.0/np.pi) * self.alpha_s(mu, loop=2))  

        return n_1

    
    def n_2(self, mu):

        '''
        The second order (NNLO) term in the number density in Eq. (61)
        of Kurkela et al. (2010). 

        :Example:
            PQCD.n_2(mu=np.linspace())

        Parameters:
        -----------
        mu : numpy.linspace
            The original mu range.

        Returns:
        --------
        n_2 : numpy.ndarray
            The second order contribution to the number density.
        '''

        lambda_bar = self.X * mu

        outer = self.n_FG(mu)
        one = 1.0
        two = 2.0*self.alpha_s(mu, loop=2)/np.pi
        three = (self.alpha_s(mu, loop=2)/np.pi)**2.0
        four = (61.0/4.0) - 11.0*np.log(2) - 0.369165*self.Nf 
        five = self.Nf*np.log(self.Nf*self.alpha_s(mu, loop=2)/np.pi) 
        six = self.beta0*np.log(lambda_bar/mu)

        n_2 = outer * (one - two - three*(four + five + six))

        return n_2

    
    # define the pressure (from Gorda et al. (2018)) 
    def pressure_FG(self, mu):

        '''
        The FG contribution to the pressure in terms of mu.

        :Example:
            PQCD.pressure_FG(mu=np.linspace())

        Parameters:
        -----------
        mu : numpy.linspace
            The original range in mu.

        Returns:
        --------
        p_FG : numpy.linspace
            The zeroth order (LO) contribution the pressure. 
        '''

        p_FG = (3.0 * mu**4.0) / (4.0 * np.pi**2.0)

        return p_FG
    
    
    def pressure_mu(self, mu, order=2):

        '''
        The pressure with respect to mu for either NLO or NNLO.
        Note: alpha_s has been used up to second order here to
        maintain the validity of the value of alpha_s at 2 GeV.

        :Example:
            PQCD.pressure_mu(mu=np.linspace(), order=1)

        Parameters:
        -----------
        mu : numpy.linspace
            The original range of mu.

        order : int
            The order at which the pressure is calculated. Default is 2,
            options are 1 and 2.

        Returns:
        --------
        pressure : numpy.ndarray
            The pressure as a function of mu.
        '''
 
        lambda_bar = self.X * mu 
        
        if order == 1:

            pressure = self.pressure_FG(mu) * (1.0 - 2.0 * \
                self.alpha_s(mu, loop=2)/np.pi)
            
            return pressure
        
        elif order == 2:
        
            first = 2.0 * self.alpha_s(mu, loop=2) / np.pi
            second = 0.303964 * self.alpha_s(mu, loop=2)**2.0 \
                * np.log(self.alpha_s(mu, loop=2))
            second_2 = self.alpha_s(mu, loop=2)**2.0 * (0.874355 \
                + 0.911891 * np.log(lambda_bar/mu))
            pressure = self.pressure_FG(mu) * (1.0 - first - \
                second - second_2)
            
            return pressure
        
    
    # define a function for the zeroth order in pressure wrt n
    def pressure_n_FG(self, n):
        
        '''
        FG pressure with respect to number density n.
        
        Parameters:
        -----------
        n : numpy.ndarray
            The baryon number density. 
            
        Returns:
        --------
        p_n_FG : numpy.ndarray
            The FG pressure as a function of density.
        '''

        p_n_FG = (3.0 / 4.0 * np.pi**2.0) * (n * np.pi**2.0 / 3.0)**(4.0/3.0)

        return p_n_FG
        
    
    # define a function for the first order in pressure wrt n 
    def pressure_n(self, n, alpha_s_mu=None):

        '''
        The pressure as a function of number density, to first
        order.

        :Example:  
            PQCD.pressure_n(n=np.linspace(), alpha_s_mu=None)

        Parameters:
        -----------
        n : numpy.linspace
            The range in number density being considered.
        
        alpha_s_mu : numpy.linspace, numpy.ndarray
            A possible different chemical potential range
            to send to alpha_s. Default is None.
            
        Returns:
        --------
        p_n : numpy.ndarray
            The pressure as a function of density, at NLO.
        '''
        
        if alpha_s_mu is not None:
            p_n = self.pressure_n_FG(n) * \
                (1.0 + 2.0 * self.alpha_s(mu=alpha_s_mu, loop=2)/(3.0 * np.pi))**(4.0) \
                * (1.0 - 2.0 * self.alpha_s(mu=alpha_s_mu, loop=2))
        else:
            raise ValueError('Supply a value of mu for alpha_s.')

        return p_n
    

    # define the function for the adiabatic speed of sound
    def cs2(self, n, mu):
        
        '''
        Calculaton of the speed of sound. Not used in this
        paper.
        
        Parameters:
        -----------
        n : numpy.ndarray
            Number density array.
            
        mu : numpy.ndarray
            Corresponding chemical 
            potential array. 
            
        Returns:
        --------
        cs2 : numpy.ndarray
            The speed of sound.
        '''

        # mu_FG again
        mu_FG = ((n*np.pi**2.0)/3.0)**(1.0/3.0)

        # determine denominator in mu
        denom = (1.0 - 2.0 * self.alpha_s(mu=mu_FG, loop=2)/np.pi)**(1.0/3.0)

        # now calculate derivative of mu wrt n
        dmudn = (1.0/3.0) * (np.pi**2.0 / (3.0 * denom))**(1.0/3.0) * n**(-2.0/3.0)

        # calculate cs2
        cs2 = (n / mu) * dmudn 
 
        return cs2
    
    
    # define the pressure at N3LO from Gorda et al. (2023)
    def pressure_n3lo(self, mu, scaled=False):
        
        '''
        A first attempt at coding the results from 
        Gorda et al. (2023) including N3LO. 
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential. 
        
        scaled : bool
            Whether the data is scaled or not. 
            Default is False. 
        
        Returns:
        --------
        p_n3lo : numpy.ndarray
            The pressure at N3LO. 
        '''

        # define constants and pieces
        lambda_bar = self.X * mu 
        c0 = -23.0 # 68% uncertainty of +/- 10.0
        
        # define constants
        c_32 = 11.0/12.0
        c_31 = -6.5968 - 3.0 * np.log(lambda_bar/(2.0*mu))
        c_30 = 5.1342 + (2.0/3.0)*c0 - 18.284 * np.log(lambda_bar/(2.0*mu)) \
            - (9.0/2.0) * np.log(lambda_bar/(2.0*mu))**2.0
        
        p_free = 3.0 * mu**4.0 / (4.0 * np.pi**2.0)
        
        # define the equation (with and without scaling)
        one = np.log(3.0 * self.alpha_s(mu,loop=2)/np.pi) + \
                3.0 * np.log(lambda_bar/(2.0*mu)) + 5.0021 
        two = c_32 * np.log(3.0*self.alpha_s(mu,loop=2)/np.pi)**2.0 + \
                c_31 * np.log(3.0*self.alpha_s(mu,loop=2)/np.pi) + c_30
        pressure = 1.0 - 2.0 * (self.alpha_s(mu, loop=2)/np.pi) - \
                3.0 * (self.alpha_s(mu, loop=2)/np.pi)**2.0 * one + 9.0 * (self.alpha_s(mu,loop=2)/np.pi)**3.0 * two
        
        if scaled is True:
            p_n3lo = pressure 
        else:
            p_n3lo = p_free * pressure

        return p_n3lo
    
    
    # define a helper function for masking arrays with nan or negative values
    @staticmethod
    def mask_array(array, neg=False, fill_value=None):

        '''
        Returns a masked array from an original array that contains 
        unwanted nan values. Can also fill values in the place of 
        the mask, if desired.

        :Example:
            PQCD.mask_array(array=mu, fill_value=0)

        Parameters:
        -----------
        array : numpy.ndarray
            The array with values to mask. 

        neg : bool
            If False, will not also mask negative values. If True,
            will check for negative values and mask using current
            fill value. 

        fill_value : int, float
            The value with which to fill the mask. Default is None.

        Returns:
        --------
        masked_array : numpy.ndarray
            The masked array with or without filled values.
        '''

        # initialize the mask and new array
        mask = np.zeros(len(array), dtype=bool)
        masked_array = np.zeros(len(array))

        # check for neg bool and mask if needed
        if neg is True:
            for i in range(len(array)):
                if array[i] < 0.0:
                    array[i] = fill_value

        # check for nan values and replace in mask
        for i in range(len(array)):
            if np.isnan(array[i]):
                mask[i] = True

        # mask the original array
        masked_array = np.ma.array(array, mask=mask)
        
        # if there were given fill values, fill the array
        if fill_value is not None:
            masked_array = np.ma.filled(masked_array, fill_value=fill_value)

        return masked_array


class Kurkela(PQCD):

    '''
    A simple class to find the roots and perform the integration
    needed for the Kurkela et al. (2010) CQM paper formulation of 
    the pressure of massless quarks. 
    
    Also converts from chemical potential to number density if
    needed.
    '''

    def __init__(self, X=2, Nf=3):

        # initialise PQCD class to use below
        self.X = X
        self.Nf = Nf
        self.pqcd_root = PQCD(X=self.X, Nf=self.Nf)

        return None
    

    def pressure(self, mu, n_mu=None, scaled=True):

        '''
        Calculation of the pressure with respect to the chemical
        potential using the method of Kurkela et al. (2010). 
        Employs the integration of the number density with respect
        to mu, using the integration limits determined by the find_roots
        function. 

        Parameters:
        -----------
        mu : numpy.ndarray
            The linspace in chemical potential with which we started.
            
        n_mu : numpy.ndarray
            The corresponding number density.

        scaled : bool
            Toggle to return either scaled pressure (wrt free quark gas)
            or to return without scaling. Default is True.
        
        Returns:
        --------
        mu_array : dict
            Dictionary of the new mu arrays spanning the integration
            limits of the pressure integration. 
 
        pressure_1 : numpy.ndarray
            The mean value of the pressure at first order for the chosen 
            chemical potential (or number density) array. 

        pressure_2 : numpy.ndarray
            The mean value of the pressure at second order for the chosen 
            chemical potential (or number density) array. 

        '''

        ### In this version, only using the second order limits 
        ### since they are higher in mu than the first order ones.
        ### A necessary condition for BUQEYE truncation errors at
        ### the present time. 
              
        # set integration limits first
        roots = self.find_roots()

        if self.X == 2:
            mu_0 = {
                "2" : roots,
            }

        # bag constant
        B = 0.0
        hbarc = 0.197327**3.0

        # set up dicts
        p0 = defaultdict(list)
        P = defaultdict(list)
        mu_array = defaultdict(list)

        # X values and orders keys for dicts
        if self.X == 2:
            mu_array_keys = ["x2n1", "x2n2"]
            mu_keys = ["2", "2"]

            # create each array for each X and order 
            # (using second order chemical potential array)
            mu_array_i = np.concatenate((np.array([mu_0["2"][1]]), mu)) 
            for i,j in zip(mu_array_keys, [0,1]):
                #mu_array[i].append(np.linspace(mu_0["2"][1], max(mu), len(mu))) # concatenate mu_0 with mu_n (OK b/c cutoff always lower)
                mu_array[i].append(mu_array_i)
                
            # first order, X=2
            for i in mu_array["x2n1"][0]:   # now we are in mu_B
                p0["x2n1"].append(self.pqcd_root.pressure_FG(i)/hbarc)
                P["x2n1"].append(-B + sc.quad((lambda x : (1.0/3.0) * \
                                self.pqcd_root.n_1(x/3.0)), 3.0 * mu_0["2"][0], 3.0 * i)[0])

            # second order, X=2
            for i in mu_array["x2n2"][0]: 
                p0["x2n2"].append(self.pqcd_root.pressure_FG(i)/hbarc)
                P["x2n2"].append(-B + sc.quad((lambda x : (1.0/3.0) * \
                                 self.pqcd_root.n_2(x/3.0)), 3.0 * mu_0["2"][1], 3.0 * i)[0])

            # class variables for p0
            #p0 = {k: v / hbarc for k, v in p0.items()}
            self.p0 = p0    #unscaled, be careful here

#             # re-set mu_array to 100 points we need in density
#             mu_array["x2n1"][0] = mu_array["x2n1"][0][1:]
#             mu_array["x2n2"][0] = mu_array["x2n2"][0][1:]

            if scaled == True:
                p_1 = np.asarray(P["x2n1"])/hbarc/np.asarray(p0["x2n1"])#[1:])
                p_2 = np.asarray(P["x2n2"])/hbarc/np.asarray(p0["x2n2"])#[1:])
                return mu_array, p_1, p_2
            else:
                p_1 = np.asarray(P["x2n1"])/hbarc#[1:])
                p_2 = np.asarray(P["x2n2"])/hbarc#[1:])
                return mu_array, p_1, p_2

        else:
            return None
        
        
    def speed_sound(self, mu, n):
        
        # speed of sound for the Kurkela EOS using
        # inversion as for the pressure 
        
        '''
        The speed of sound calculation. Not used in 
        the paper. 
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The chemical potential. 
            
        n : numpy.ndarray
            The number density.
        
        Returns:
        --------
        cs2 : numpy.ndarray
            The speed of sound. 
        '''
        import numdifftools as ndt
        
        # terms
        one = n/mu
        dndmu = ndt.Derivative(self.pqcd_root.n_2, n=1)
        two = dndmu(mu)
        
        cs2 = one * (two)**(-1.0)   # handles inverse of derivative
        
        return cs2
        

    def uncertainties(self, mu, n_orders=3, kernel=None, test=None):

        '''
        Calculation of the truncation error bands for the pQCD EOS,
        using the Kurkela et al. (2010) formulation for the pressure.
        This function uses techniques from the gsum package. 

        Parameters:
        -----------
        mu : numpy.ndarray
            The linspace of chemical potential needed.
        
        n_orders : int
            The highest order to which the pressure EOS is calculated.
            
        kernel : obj
            The kernel needed for the interpolation and truncation GP.
            Can be fed in from the outside to change parameters.
            
        test : numpy.ndarray
            Testing array. 
        
        Returns:
        --------
        data : numpy.ndarray
            The data array.
        
        self.coeffs : numpy.ndarray
            The values of the coefficents at the chemical potential mu.
        
        std_trunc : numpy.ndarray
            The arrays of truncation errors per each order.

        '''

#         # grab the correct mu_array from the pressure
#         mu_array, _, _ = self.pressure(mu, scaled=True)

#         # reshape the mu_array to what we need
#         mu = mu_array["x2n2"][0]    # need the quark chemical potential
        mu_arr = mu[:, None]
        self.mu_arr = mu_arr          # for use outside

        # orders
        orders = np.arange(0, n_orders)

        # construct the mask
        self.mask = self.gp_mask(mu)

        # construct the needed data and coefficients
        if test is not None:
            coeffs_all = test
        else:
            coeffs_all = np.array([self.c_0(mu), self.c_1(mu), self.c_2(mu)]).T
        data_all = gm.partials(coeffs_all, ratio=self.expQ(mu_arr), \
                               ref=self.yref(mu_arr), orders=[0,1,2])

        # shape for correct format
        self.coeffs = coeffs_all[:, :n_orders]
        data = data_all[:, :n_orders]

        # call the kernel function 
        if kernel is None:
            self.kernel = self.gp_kernel()
        else:
            self.kernel = kernel

        # set up the truncation GP
        trunc_gp = gm.TruncationGP(kernel=self.kernel, ref=self.yref, \
                            ratio=self.expQ, disp=0, df=3, scale=1, optimizer=None) # disp is variance of the mean
        trunc_gp.fit(mu_arr[self.mask], data[self.mask], orders=orders)

        std_trunc = np.zeros([len(mu_arr), n_orders])
        for i, n in enumerate(orders):
            # Only get the uncertainty due to truncation (kind='trunc')
            _, std_trunc[:,n] = trunc_gp.predict(mu_arr, order=n, return_std=True, kind='trunc')

        return data, self.coeffs, std_trunc
    
    
    def inversion(self, n_mu=None):
        
        '''
        Function to invert n(mu) to obtain mu(n). 
        
        Parameters:
        -----------
        n_mu : numpy.array
            Linspace over n_q for the inversion.
        
        Returns:
        --------
        f_mu_1_result, f_mu_2_result : numpy.array
            The two arrays corresponding to the inverted function values
            for mu(n^(1)) and mu(n^(2)). 
        '''
        
        # write the root finding function
        def f_mu_1(n, guess):  
            return opt.fsolve(lambda mu : n - self.pqcd_root.n_1(mu), x0 = guess)[0]

        def f_mu_2(n, guess):  
            return opt.fsolve(lambda mu : n - self.pqcd_root.n_2(mu), x0 = guess)[0]

        # call the function
        f_mu_1_result = []
        f_mu_2_result = []
        
        if n_mu is None:
            n_mu = np.linspace(0.01, 0.8, 1000)  # n_q right now -> 3*n_B = (0.0, 1.25) GeV^3
        else:
            n_mu = n_mu

        # invert to get mu(n^1) and mu(n^2)
        for i in n_mu:
            f_mu_1_result.append(f_mu_1(i, guess=2.0))   # these results will be in muB/3 (because of n(mu) function)
            f_mu_2_result.append(f_mu_2(i, guess=2.0))
            
        # convert to n_B here to not make mistake later
        self.nB = n_mu/3.0   # n_q/3

        # convert the mu arrays over (still muB/3)
        f_mu_1_result = np.asarray(f_mu_1_result)
        f_mu_2_result = np.asarray(f_mu_2_result)
        
        return n_mu, f_mu_1_result, f_mu_2_result
        
    
    def gp_interpolation(self, mu, kernel=None, center=0.0, sd=1.0):

        '''
        The function responsible for fitting the coefficients with a GP
        and predicting at new points. This information will be used in 
        constructing our truncated GP in the function 'Uncertainties'. 

        Parameters:
        -----------
        mu : numpy.ndarray
            The chemical potential linspace needed.
            
        kernel : obj
            The kernel needed for the interpolation GP. Can be fed in 
            from the outside for specific parameter alterations.
            
        center : float
            Value for the center of the prior. 
        
        sd : float
            The scale of the prior. 

        Returns:
        --------
        pred : numpy.ndarray
            An array of predictions from the GP.

        std : numpy.ndarray
            The standard deviation at the points in 'pred'.

        underlying_std : numpy.ndarray
            The underlying standard deviation of the GP.
        '''

#         # grab the correct mu_array from the pressure
#         mu_array, _, _ = self.pressure(mu, scaled=True)

        # reshape the mu_array to what we need
        #mu = mu_array["x2n2"][0]    # need the quark chemical potential
        mu_arr = mu[:, None]

        # interpolate the coefficents using GPs and gsum 
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
        self.mask = self.gp_mask(mu)

        # Set up gp objects with fixed mean and standard deviation 
        if kernel is None:
            self.kernel = self.gp_kernel()  # in case kernel has not been yet called
        else:
            self.kernel = kernel
            self.gp_interp = gm.ConjugateGaussianProcess(
                kernel=self.kernel, center=center, disp=0, df=3.0, scale=sd, nugget=0) 
        
        # fit and predict using the interpolated GP
        self.gp_interp.fit(mu_arr[self.mask], self.coeffs[self.mask])
        pred, std = self.gp_interp.predict(mu_arr, return_std=True)
        underlying_std = np.sqrt(self.gp_interp.cov_factor_)
        
        # print the kernel parameters for viewing outside the function
        print(self.gp_interp.kernel_)
        print(self.gp_interp.cov_factor_)

        return pred, std, underlying_std

    
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
        
        # mask for values above 40*n0 only 
        low_bound = next(i for i, val in enumerate(mu)
                                  if val > 0.88616447)
        mu_mask = mu[low_bound:]
        mask_true = np.array([(i) % 25 == 0 for i in range(len(mu_mask))])
        
        # concatenate with a mask over the other elements of mu before low_bound
        mask_false = np.full((1,len(mu[:low_bound])), fill_value=False)
        self.mask = np.concatenate((mask_false[0], mask_true))
        
        # old mask
        #self.mask = np.array([(i-3) % 25 == 0 for i in range(len(mu))]) #np.array([(i-6) % 25 == 0 for i in range(len(mu))])

        return self.mask 
    
    
    def gp_kernel(self):

        '''
        The kernel that we will use both for interpolating the 
        coefficients and for predicting the truncation error bands.

        Parameters:
        -----------
        None.

        Returns:
        --------
        self.kernel : sklearn object
            The kernel needed for the GPs in both 'uncertainties' and 
            'gp_interpolation'. 
        '''

        self.ls = 0.496    # starting guess; can get really close if we set 0.25 and fix it
        self.sd = 0.2    # makes a difference on the band of the regression curve for c_2 
        self.center = 0
        self.nugget = 1e-10  # nugget goes to Cholesky decomp, not the kernel (kernel has own nugget)
        self.kernel = RBF(length_scale=self.ls, length_scale_bounds='fixed') + \
        WhiteKernel(noise_level=self.nugget, noise_level_bounds='fixed') # letting this vary 

        return self.kernel
    
    
    def c_0(self, mu):

        '''
        LO coefficent for the pQCD EOS using Kurkela formalism.

        Parameters:
        -----------
        mu : numpy.ndarray
            The chemical potential for the pQCD EOS.

        Returns:
        --------
        np.ones(len(mu)) : numpy.ndarray
            The leading order coefficient. 
        '''

        return np.ones(len(mu))
    
    def c_1(self, mu):

        '''
        The NLO coefficient for the pQCD EOS using Kurkela
        formalism. 
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential array.
        
        Returns:
        --------
        The value of the coefficient. 
        '''

        # reshape the mu array for yref
        mu_arr = mu[:, None]

        # calculate the pressure 
        _, p_1, _ = self.pressure(mu, scaled=False)
        
        # take only needed pressure (fix later)
        p_1 = p_1[1:]

        one = p_1/self.yref(mu_arr)
        two = self.c_0(mu)

        return (one - two)/self.pqcd_root.alpha_s(mu, loop=2)
    
    
    def c_2(self, mu):

        '''
        The N2LO coefficent for the pQCD EOS using Kurkela
        formalism. 
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential. 
        
        Returns:
        --------
        The value of the coefficient. 
        '''

        # reshape the mu array again
        mu_arr = mu[:, None]

        # calculate the pressure
        _, _, p_2 = self.pressure(mu, scaled=False)
        
        # take only needed pressure (fix later)
        p_2 = p_2[1:]

        one = p_2/self.yref(mu_arr)
        two = self.c_1(mu) * self.pqcd_root.alpha_s(mu, loop=2)
        three = self.c_0(mu)

        return (one - two - three)/np.square(self.pqcd_root.alpha_s(mu,loop=2))


    def yref(self, mu):

        '''
        The reference for the expansion. 
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential.
        
        Returns:
        ---------
        yref : numpy.ndarray
            The values of yref at the given
            chemical potentials. 
        '''
        
        # scaling
        hbarc = 0.197327**3.0

        yref = ((3.0 * mu**4.0) / (4.0 * np.pi**2.0))
        yref = yref[:,0]/hbarc

        return yref
    

    def expQ(self, mu):

        '''
        The expansion parameter, Q, of the pQCD EOS using
        the Kurkela formalism. 
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potentials. 
            
        Returns:
        ---------
        The value of expQ at the chosen potentials. 
        '''

        return self.pqcd_root.alpha_s(mu, loop=2)[:,0]


    def find_roots(self):

        '''
        Function to determine the value of the chemical potential
        when the number density reaches zero, so that we can define
        the limits of our integration for the pressure.

        Parameters:
        -----------
        None.

        Returns:
        --------
        [root1, root2] : numpy.array
            The roots of the first and second order number
            densities in the chemical potential. 
        '''

        # set the linspace needed to solve precisely for the root
        mu_root = np.linspace(0.2, 2.5, 10000)

        # set up the root finder for both orders in n
        root_first = np.vstack((mu_root, self.n_first(mu_root)))
        index1 = np.where(root_first[1,:] > 0.0)[0][0]
        root1 = mu_root[index1]
        
        root_second = np.vstack((mu_root, self.n_second(mu_root)))
        index2 = np.where((root_second[1,:] > 0.0))[0][0]
        root2 = mu_root[index2]

        # print statement for roots if desired
        # print('The roots found for X = {} are:'.format(self.X))
        # print('n_1 = {}, n_2 = {}'.format(root1, root2))
        
        return [root1, root2]
    

    def n_first(self, mu):

        '''The first order in the number density, to be solved
        to find the root.
        
        Parameters:
        ------------
        mu : numpy.ndarray
            The quark chemical potentials. 
            
        Returns:
        --------
        n : numpy.ndarray
            The number density array. 
        '''

        # set up the function
        n = self.pqcd_root.n_1(mu)/self.pqcd_root.n_FG(mu)

        return n
    

    def n_second(self, mu):

        '''The second order in the number density, to be solved
        to find the root.
        
        Parameters:
        ------------
        mu : numpy.ndarray
            The quark chemical potentials. 
            
        Returns:
        --------
        n : numpy.ndarray
            The number density. 
        '''

        # set up the function
        n = self.pqcd_root.n_2(mu)/self.pqcd_root.n_FG(mu)

        return n
