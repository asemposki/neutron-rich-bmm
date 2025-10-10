import numpy as np

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