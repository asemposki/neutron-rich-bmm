##############################################
# Symbolic script for cs2(mu)
# Author: Alexandra Semposki
# Date: 13 December 2023
##############################################

import numpy as np
import sympy as sy

from sympy.abc import mu, alpha, n, epsilon, X, a, b, c, d

N_f = sy.symbols("N_f")
log_mu = sy.symbols("log(mu)")
lambda_bar = X * mu

def alpha_s(mu):

    return sy.symbols("alpha_s", cls=sy.Function)(mu)


def alpha_s_analytic(mu):

    beta0 = sy.symbols("beta0")
    lambda_MS = sy.symbols("lambda_MS")
    L = sy.log(X**2 * mu**2 / lambda_MS**2)

    al = (4.0 * sy.pi) / (beta0 * L)

    return al

def alpha_log(log_mu):
    return sy.symbols("alpha_s", cls=sy.Function)(log_mu)

def pressure_FG_log(log_mu):
    return a * sy.exp(log_mu)**4

def pressure_log(log_mu):
    zeroth = 1.0
    first = b * alpha_log(log_mu)
    second = (c + d * sy.log(alpha_log(log_mu))) * alpha_log(log_mu)**2
    pres = pressure_FG_log(log_mu) * (zeroth + first + second)

    return pres

def work_L(lambda_bar):
    lambda_MS = sy.symbols("lambda_MS")
    return sy.log(lambda_bar**2 / lambda_MS**2)


def pressure_FG(mu):
    return a * mu**4
 
    
def pressure(mu):
    
    # first = 2 * alpha_s(mu) / sy.pi
    # second = 0.303964 * alpha_s(mu)**2 \
    #     * sy.log(alpha_s(mu))
    # second_2 = alpha_s(mu)**2 * (0.874355 \
    #     + 0.911891 * sy.log(lambda_bar/mu))
    # pres = pressure_FG(mu) * (1 - first - \
    #     second - second_2)

    zeroth = 1.0
    first = b * alpha_s(mu)
    second = (c + d * sy.log(alpha_s(mu))) * alpha_s(mu)**2
    pres = pressure_FG(mu) * (zeroth + first + second)

    return pres

def main():

    pressure(mu)
    print(pressure(mu))

    num_dens = sy.diff(pressure(mu),mu).simplify()
    print(num_dens)

    d2Pdmu2 = sy.diff(num_dens, mu).simplify()
    print(d2Pdmu2)

    speed = num_dens/(mu * d2Pdmu2)
    print(speed)

    speed2 = speed

    # do we need this line?
    speed_subbed = speed2.replace(sy.Derivative(alpha_s(mu), mu), alpha_s(mu)) #speed2.subs(sy.Derivative(alpha_s, mu), alpha_s**2) #deprecated to use strings like this so let's try to use symbol or function
    print(speed_subbed)

if __name__== "__main__":
    main()