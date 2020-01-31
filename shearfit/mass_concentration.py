import numpy as np

'''
This module contains a collection of concentration-mass relations as given in the literature. All
the functions below take a halo mass as an argument, and will return a predicted concentration.
'''

def child2018(m200c, fit='nfw_stack'):
    """
    Computes the predicted halo concentration, given a M_200c halo mass, using the 
    c-M relation of Child et.al. 2018.

    Parameters
    ----------
    m200c : float 
        The halo's mass :math:`M_{200c}` in units of :math:`M_{\\odot} h^{-1}`.
    fit : string
        The set of fit parameters to use (Table 1 in Child+2018); options are
        `'all'`, `'relaxed'`, `'nfw_stack'`, or `'einasto_stack'`.
    
    Returns
    -------
    float array
        The concentration parameter :math:`c_{200c}`, and the associated error, :math:`c_{200c}/3`.
    """
    
    p_all = {'m':-0.10, 'A':3.44, 'b':430.49, 'c0':3.19}
    p_relax = {'m':-0.09, 'A':2.88, 'b':1644.53, 'c0':3.54}
    p_nfw_stack = {'m':-0.07, 'A':4.61, 'b':638.65, 'c0':3.59}
    p_einasto_stack = {'m':-0.01, 'A':63.2, 'b':431.48, 'c0':3.36}
    p = {'all':p_all, 'relax':p_relax, 'nfw_stack':p_nfw_stack, 'einasto_stack':p_einasto_stack}
    
    m = p[fit]['m']
    A = p[fit]['A']
    b = p[fit]['b']
    c0 = p[fit]['c0']
    mstar = 1e12
    
    c200c = A * ((m200c/mstar/b)**m * (1 + m200c/mstar/b)**(-m) -1) + c0
    c_err = c200c/3

    return [c200c, c_err]
