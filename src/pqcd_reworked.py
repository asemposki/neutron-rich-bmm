###########################################################
# pQCD class for working with N2LO, N3LO results from 
# Gorda et al. (2021, 2023).
# Author : Alexandra Semposki
# Date : 31 July 2023 (Happy Birthday Harry Potter)
# ##########################################################

# imports
import numpy as np
import numdifftools as ndt
import scipy.optimize as opt

# set up class
class PQCD:

    def __init__(self, X=1, Nf=2):

        '''
        Initialize the class so that mu and lambda_bar
        have been set and constants have been defined for
        the rest of the code.

        :Example:
            PQCD(X=1, Nf=2)

        Parameters:
        -----------
        
        X : int
            The value of the coefficient multiplying mu
            to determine the renormalization scale. Default is 1,
            as in Gorda et al. (2023).

        Nf : int
            The number of different quark flavours being 
            considered. Default is 2, for SNM. NSM is 3.

        Returns:
        --------
        None.
        '''

        # initialize constants
        self.X = X
        self.Nc = 3
        self.Nf = Nf
        self.Ca = self.Nc
        self.Cf = (self.Nc**2.0 - 1.0)/(2.0*self.Nc)
        self.lambda_MS = 0.38 #[GeV]

        self.beta0 = (11.0*self.Ca - 2.0*self.Nf)/3.0   # for alpha_s
        
        # QCD constants 
        self.b0 = (11.0/6.0) * self.Nc * self.Nf**(-1.0) - (1.0/3.0)
        self.b1 = (17.0/12.0) * self.Nc**2 * self.Nf**(-2) + \
        (self.Nc**(-1.0) * self.Nf**(-1.0))/8.0 - \
        (13.0/24.0) * self.Nc * self.Nf**(-1.0)
        self.delta = -0.85638
        
        # pressure constants
        self.dA = 8.0
        self.a11 = -0.75 * self.dA * self.Nc**(-1.0)
        self.a21 = -(3.0/8.0) * self.dA * self.Nc**(-1.0)  
        self.a22 = self.b0 * self.a11 
        self.a23 = self.dA * (((11.0/6.0) - (np.pi**2)/8.0 - (5.0 * np.log(2))/4.0 + \
                    np.log(2)**2  - (3.0/16.0)*self.delta) * self.Nc**(-1.0) - \
                    (415.0/192.0)*self.Nf**(-1.0) - \
                    (51.0/64.0)*self.Nc**(-2) * self.Nf**(-1.0))
                
        return None
    
    
    def n_convert_mu(self, density):
        
        '''
        The function that converts from a desired number 
        density to quark chemical potential. 
        
        Parameters:
        -----------
        density : numpy.linspace
            The density array.
            
        Returns:
        --------
        mu_FG : numpy.ndarray
            The FG quark chemical potential. 
        
        mu_n : numpy.ndarray
            The quark chemical potential array.
        '''
        
        n0 = 0.164
        
        # take the density and invert to obtain the corresponding mu
        n_q = density*3.0  # n_q [fm^-3]
        
        # convert to GeV^3 for mu_q
        conversion_fm3 = ((1000.0)**(3.0))/((197.33)**(3.0)) # [fm^-3]  (do the opposite of this)
        n_q = n_q/conversion_fm3
        
        # invert to obtain mu_n and mu_FG
        _, mu_n, mu_FG = self.inversion(n_mu=n_q)
        
        return mu_FG, mu_n
    
    
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
            The values of alpha_s at the order requested.
        '''

        lambda_bar = 2. * self.X * mu   # X range [1/2, 2]

        # renormalization scale
        ell = np.log((lambda_bar**2.0)/self.lambda_MS**2.0)

        if (loop == 1):
            beta1 = 0.0
            # running coupling constant
            alpha_s = ((4.0*np.pi)/(self.beta0*ell))
        elif (loop == 2):
            beta1 = (17.0/3.0)*self.Ca**2.0 - self.Cf*self.Nf - (5.0/3.0)*(self.Ca*self.Nf) 
            # running coupling constant
            alpha_s = ((4.0*np.pi)/(self.beta0*ell)) * (1.0 - (2.0*beta1*np.log(ell))/(ell*self.beta0**2.0))
        else:
            raise ValueError('Loop selected must be 1 or 2.')

        return alpha_s
    

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

        p_FG = (self.Nf * self.Nc * mu**4.0) / (12.0 * np.pi**2.0)  # extremely general form

        return p_FG
    

    def pressure_mu(self, mu, order=2):

        '''
        The pressure with respect to mu for NLO or N2LO.
        Note: alpha_s has been used up to second order here to
        maintain the validity of the value of alpha_s at 2 GeV.
        From Gorda et al. (2023), supplemental material.

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
 
        lambda_bar = 2. * self.X * mu     # X range [1/2, 2]
    
        # log terms
        logQ = np.log(self.Nf*self.alpha_s(mu, loop=2) / np.pi)
        logL = np.log(lambda_bar/(2.0*mu))
            
        if order == 1:
            pressure = self.pressure_FG(mu) * (1.0 + self.a11*(self.alpha_s(mu, loop=2)/np.pi))
        
        elif order == 2:
            
            # N2LO terms
            first = self.a21 * logQ
            second = self.a22 * logL
            third = self.a23
            
            # full expression
            pressure_NLO = self.a11*(self.alpha_s(mu, loop=2)/np.pi)
            pressure_N2LO = ((first + second + third)*self.Nf \
                             *(self.alpha_s(mu,loop=2)/np.pi)**2)
            
            pressure = self.pressure_FG(mu) * (1.0 + pressure_NLO + pressure_N2LO)
            
        return pressure
    
    
    def pressure_old(self, mu, order=2):
        
        '''
        Old version of the pressure, using Nf=3
        implicitly. Not used in the paper.
        
        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential.
            
        Returns:
        --------
        pressure : numpy.ndarray
            The value of the pressure at the
            chemical potentials. 
        '''
        
        lambda_bar = 2.0 * self.X * mu
        
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
 

    def n_mu(self, mu):

        '''
        Simple first derivative of the pressure (up to second
        order in P(mu) equation) with respect to the chemical 
        potential to obtain the number density with respect to mu.
        
        NOTE: this derivative is to second order in P, so the
        equation will be: n(mu) = dP0/dmu + dP1/mu + dP2/mu.

        Parameters:
        -----------
        mu : numpy.ndarray
            The quark chemical potential. 

        Returns:
        --------
        n : numpy.ndarray
            The array of the number density with respect
            to the chemical potential.
        '''

        n_mu = ndt.Derivative(self.pressure_mu, step=0.00001, method='central')
        n = n_mu(mu)

        return n
    

    def n_FG_mu(self, mu):

        '''
        Free quark number density calculation from P_FG(mu)
        derivative. Note that this will yield a different n 
        value for the same mu value as n_mu yields, so 
        scaling must be taken into account.

        Parameters:
        -----------
        mu : numpy.ndarray
            The input chemical potential.

        Returns:
        --------
        n_FG : numpy.ndarray
            The result of the number density at the input 
            chemical potential.
        '''

        n_FG = self.Nf * mu**3.0 / (np.pi**2.0)

        return n_FG
    
    
    def inversion(self, n_mu=None):
        
        '''
        Function to invert n(mu) to obtain mu(n). 
        
        Parameters:
        -----------
        n_mu : numpy.array
            Linspace over n_q for the inversion.
        
        Returns:
        --------
        n_mu : numpy.ndarray
            The number density. 

        f_mu_2_new : numpy.array
            The array corresponding to the inverted function values
            for mu(n).
        
        f_mu_FG_new : numpy.ndarray
            The values of the FG chemical potential. 
        '''
        
        # write the root finding function
        def f_mu_2(n, guess):  
            return opt.fsolve(lambda mu : n - self.n_mu(mu), x0 = guess)[0]
        
        # FG chemical potential (mu_FG)
        def f_mu_FG(n, guess):
            return opt.fsolve(lambda mu_FG : n - self.n_FG_mu(mu_FG), x0 = guess, xtol=1e-10)[0]
        
        if n_mu is None:
            n_mu = np.linspace(0.01, 0.8, 1000)  # n_q right now -> 3*n_B = (0.0, 1.25) GeV^3
        
        # invert to get mu(n^2)
        f_mu_2_new = []
        f_mu_FG_new = []
        for i in n_mu:
            f_mu_2_new.append(f_mu_2(i, guess=2.0))
            f_mu_FG_new.append(f_mu_FG(i, guess=2.0))
            
        # convert to n_B here to not make mistake later
        self.nB = n_mu/3.0   # n_q/3

        # convert the mu arrays over (still muB/3 -> mu_q)
        f_mu_2_new = np.asarray(f_mu_2_new)
        self.f_mu_FG_new = np.asarray(f_mu_FG_new)

        return n_mu, f_mu_2_new, self.f_mu_FG_new
    
    # adding these in for now
    def yref(self, x):
        
        '''
        The function to evaluate yref from pQCD.
        
        Parameters:
        -----------
        x : numpy.array
            The quark chemical potential.
            
        Returns:
        --------
        yref : numpy.2darray
            The [:,None] array for yref.
        ''' 
        
        yref = ((self.Nf * self.Nc * x**4.0) / (12.0 * np.pi**2.0))  # FG pressure for any flavour, colour
        yref = yref[:,0]
        return yref
    
    
    def expQ(self, x): 
        
        '''
        The expansion parameter function for pQCD. 
        Default here is Nf * alpha_s / pi, but another may
        be substituted.
        
        Parameters:
        -----------
        x : numpy.array
            The quark chemical potential.
        
        Returns:
        --------
        The value of the expansion parameter at each
        point in the quark chemical potential array.
        '''
        return (self.Nf * self.alpha_s(x, loop=2)/np.pi)[:,0]
        #return self.alpha_s(x, loop=2)[:,0]  # alpha_s only
        
    
    
    def c_0(self, x):
        
        '''
        The c0 coefficient of pQCD. 
        
        Parameters:
        -----------
        x : numpy.array
            The quark chemical potential.
        
        Returns:
        --------
        The value of the coefficient at each point
        in the array.
        '''
        return np.ones(len(x))
    
    
    def c_1(self, x):
        
        '''
        The c1 coefficient of pQCD. 
        
        Parameters:
        -----------
        x : numpy.array
            The quark chemical potential.
        
        Returns:
        --------
        The value of the coefficient at each point
        in the array.
        '''
        return (self.a11*np.ones(len(x))/self.Nf)
        #return (self.a11/np.pi)*np.ones(len(x))  # alpha_s only
        
        
    
    def c_2(self, x):
        
        '''
        The c2 coefficient of pQCD. 
        
        Parameters:
        -----------
        x : numpy.array
            The quark chemical potential.
        
        Returns:
        --------
        The value of the coefficient at each point
        in the array.
        '''
        lambda_bar = 2.0 * self.X * x
        one = self.a21 * np.log(self.Nf * self.alpha_s(x, loop=2)/np.pi)
        two = self.a22 * np.log(lambda_bar/(2.0*x))
        three = self.a23
        return (one + two + three)/self.Nf
        #return (one + two + three) * self.Nf/(np.pi**2)  # alpha_s only
  
    
    def mu_1(self, mu_FG): 
        
        '''
        The mu1 term of pQCD. 
        
        Parameters:
        -----------
        mu_FG : numpy.array
            The FG quark chemical potential.
        
        Returns:
        --------
        The value of mu1 at each point
        in the array.
        '''
        
        mU_FG = mu_FG[:, None]
        numerator = self.c_1(mu_FG) * self.expQ(mU_FG) * (self.Nf * mu_FG**3.0/(np.pi**2.0))
        #numerator = self.c_1(mu_FG) * self.alpha_s(mu_FG, loop=2) * self.n_FG_mu(mu_FG) #checked 
        denominator = self.c_0(mu_FG) * (self.Nf * 3.0 * mu_FG**2.0 / (np.pi**2.0)) #checked
        return -numerator/denominator
    
    
    def mu_2(self, mu_FG): 
        
        '''
        The mu2 term of pQCD. 
        
        Parameters:
        -----------
        mu_FG : numpy.array
            The FG quark chemical potential.
        
        Returns:
        --------
        The value of mu2 at each point
        in the array.
        '''
        
        mU_FG = mu_FG[:, None]
        self.derivalpha = - self.beta0 * self.alpha_s(mu_FG, loop=2)**2.0 / (2.0 * mu_FG * np.pi)
        self.derivQ = self.derivalpha * (self.expQ(mU_FG)/self.alpha_s(mu_FG, loop=2))
        secderivP0 = self.Nf * 3.0 * mu_FG**2.0 / (np.pi**2.0)
        
        first = 0.5 * self.c_0(mu_FG) * self.mu_1(mu_FG)**2.0 * (self.Nf * 6.0 * mu_FG / (np.pi**2.0)) #checked
        #third = self.c_1(mu_FG) * derivalpha * self.pressure_FG(mu_FG) #checked
        third = self.c_1(mu_FG) * self.derivQ * self.pressure_FG(mu_FG)
        
      #  second = self.mu_1(mu_FG) * self.c_1(mu_FG) * self.alpha_s(mu_FG, loop=2) * secderivP0 #checked
        second = self.mu_1(mu_FG) * self.c_1(mu_FG) * self.expQ(mU_FG) * secderivP0
      #  fourth = self.c_2(mu_FG) * self.alpha_s(mu_FG, loop=2)**2.0 * (self.Nf * mu_FG**3.0 / (np.pi**2.0)) #checked
        fourth = self.c_2(mu_FG) * self.expQ(mU_FG)**2.0 * (self.n_FG_mu(mu_FG))
        
        mu2 = -(first + second + third + fourth)/(self.c_0(mu_FG) * secderivP0)
        
        return mu2
    
    
    # pressure equation for P(n) using KLW inversion
    def pressure_KLW(self, mu_FG):
        
        '''
        The pressure equation written out for the KLW
        inversion. 
        
        Parameters:
        -----------
        mu_FG : numpy.array
            The FG quark chemical potential.
            
        Returns:
        --------
        pressure_n : dict
            The values of the pressure from the KLW
            inversion at LO, NLO, and N2LO.
        '''
        
        # mu_FG assignment
        mU_FG = mu_FG[:,None]
        
        pressure_0 = self.c_0(mu_FG) * self.yref(mU_FG)
        
        pressure_1 = self.c_1(mu_FG)*self.expQ(mU_FG)*self.yref(mU_FG) + \
        self.mu_1(mu_FG) * self.c_0(mu_FG) * self.n_FG_mu(mu_FG)
        
        pressure_2 = self.c_2(mu_FG)*self.expQ(mU_FG)**2.0*self.yref(mU_FG) + \
        (self.mu_2(mu_FG) * self.c_0(mu_FG) * self.n_FG_mu(mu_FG) + \
         0.5 * self.mu_1(mu_FG)**2.0 * (self.Nf * 3.0 * mu_FG**2.0 / np.pi**2.0) + \
         self.mu_1(mu_FG) * self.c_1(mu_FG) * self.n_FG_mu(mu_FG) * self.expQ(mU_FG))
        
        # organise so each order contains the last
        pressure_LO = pressure_0
        pressure_NLO = pressure_0 + pressure_1
        pressure_N2LO = pressure_0 + pressure_1 + pressure_2
        
        # stash in dict
        pressure_n = {
            "LO": pressure_LO,
            "NLO": pressure_NLO,
            "N2LO": pressure_N2LO
        }
        
        return pressure_n
    
    
    # create function for the pQCD energy density anchor points 
    # (samples assumed format: [dens, samples])
    def anchor_point_edens(self, samples, anchor):

        # get density value from anchor [fm^-3]
        self.n_anchor = anchor

        # use this to calculate chemical potential
        n_q = self.n_anchor * 3.0  # n_q [fm^-3]

        # convert to GeV^3 for mu_q
        conversion_fm3 = ((1000.0)**(3.0))/((197.33)**(3.0)) # [fm^-3]  (do the opposite of this)
        n_q = n_q/conversion_fm3  # [GeV^3]

        # invert to get mu
        _, mu_n, _ = self.inversion(n_mu=n_q.reshape(-1,1))  # [GeV] # these are quark chemical potentials

        # take mu and get MeV (and baryon) version
        mu = mu_n * 1e3
        
        # now calculate the edens array using these pieces (assuming last entry is anchor point)
        edens_0_array = np.zeros([len(samples.T)])
        for i in range(len(samples.T)):
            edens_0_array[i] = 3.0 * self.n_anchor * mu - samples[-1,i]  # needs to be quark version, but MeV/fm^3!

        return edens_0_array


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
    
    
# set up a derivative class from PQCD
class PQCDDens(PQCD):
    
    def __init__(self, nb, X=1, Nf=2):
        
        super().__init__(X, Nf)      
        return None
    
    
    # number density version
    def yref_dens(self, nb):
        
        # calculate mu_FG in terms of nb
        mu_FG = (3.0 * np.pi**2.0 * 3.0 * nb * (197.33/1000.)**3.0 / (self.Nc * self.Nf))**(1.0/3.0)
    #    mU_FG = mu_FG[:,None]
       
        # calculate yref
        yref = ((self.Nf * self.Nc * mu_FG**4.0) / (12.0 * np.pi**2.0))  # FG pressure for any flavour, colour
        yref = yref[:,0]
        return yref
    
    
    # adding as a function of number density
    def expQ_dens(self, nb):
        
        # calculate mu_FG in terms of nb
        mu_FG = (3.0 * np.pi**2.0 * 3.0 * nb * (197.33/1000.)**3.0 / (self.Nc * self.Nf))**(1.0/3.0)
      #  mU_FG = mu_FG[:,None]
        
        return (self.Nf * self.alpha_s(mu_FG, loop=2)/np.pi)[:,0]

    
    # directly function of number density
    def c_0_n(self, nb):
        return np.ones(len(nb))
    
    
    # purely function of number density
    def c_1_n(self, nb):
        return (2.0/(3.0 * self.Nf))*np.ones(len(nb))
         
            
    def c_2_n(self, nb):
        
        # calculate mu_FG in terms of nb
        mu_FG = (3.0 * np.pi**2.0 * 3.0 * nb * (197.33/1000.)**3.0 / (self.Nc * self.Nf))**(1.0/3.0)
        
        one = 8.0 / (9.0 * self.Nf**2.0)
        two = self.c_2(mu_FG) / 3.0
        three = self.beta0 / (3.0 * self.Nf**2.0)
        
        return (one - two - three)
    
    
    # writing it all in terms of the full paper expression instead of perturbative mu
    # this should equal the pressure wrt mu_FG for the same calculation in the mean
    # curve; check this to see if we're correct!
    def pressure_KLW_check(self, nb):
        
        nB = nb[:,None]
        
        pressure_0 = self.yref_dens(nB)
        
        pressure_1 = (2.0/(3.0*self.Nf)) * self.expQ_dens(nB) * self.yref_dens(nB)
        
        pressure_2 = self.yref_dens(nB) * self.expQ_dens(nB)**2.0 * self.c_2_n(nb)
        
        # organise so each order contains the last
        pressure_LO = pressure_0
        pressure_NLO = pressure_0 + pressure_1
        pressure_N2LO = pressure_0 + pressure_1 + pressure_2
        
        # stash in dict
        pressure_n_check = {
            "LO": pressure_LO,
            "NLO": pressure_NLO,
            "N2LO": pressure_N2LO
        }
        
        return pressure_n_check