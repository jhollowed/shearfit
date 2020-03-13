import numpy as np
from colossus.cosmology import cosmology as colcos
from colossus.halo.concentration import concentration as mass_conc

'''
This module contains a collection of concentration-mass relations as given in the literature. All
the functions below take a halo mass as an argument, and will return a predicted concentration.
'''

def child2018(m200c_h, z, cosmo, fit='nfw_stack'):
    """
    Computes the predicted halo concentration, given a M_200c halo mass, using the 
    c-M relation of Child et.al. 2018, as implemented in COLOSSUS by Diemer.

    Parameters
    ----------
    m200c : float 
        The halo's mass :math:`M_{200c}` in units of :math:`M_{\\odot} h^{-1}`.
    z : float
        The halo redshift
    cosmo : `astropy` `cosmology` object instance
        The cosmology to use in computing `M_star`
    fit : string, optional
        The set of fit parameters to use (Table 1 in Child+2018); options are
        `'individual_all'`, `'individual_relaxed'`, `'nfw_stack'`, or `'einasto_stack'`.
        Defaults to `'individual_all'`.
    
    Returns
    -------
    float array
        The concentration parameter :math:`c_{200c}`, and the associated error, :math:`c_{200c}/3`.
    """
    
    # draw a concentration from gaussian with scale and location defined by Child+2018
    cosmo_colossus = colcos.setCosmology('OuterRim',
                     {'Om0':cosmo.Om0, 'Ob0':cosmo.Ob0, 'H0':cosmo.H0.value, 'sigma8':0.8, 
                      'ns':0.963, 'relspecies':False})
    c200c = mass_conc(m200c_h, '200c', z, model='child18')
    c_err = c200c/3

    return [c200c, c_err]
