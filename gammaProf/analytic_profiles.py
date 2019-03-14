import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7 as cosmo
import pdb

'''
This module contains a collection of ananlytic halo profile forms (currently just NFW), which 
return a prediction for the induced source tangential shear by the lens, as a function of halocentric 
projected radius. Specifically, the scaled tang. shear ΔΣ = γ_t * Σ_c is returned, which is a constant
for any identical lens, regardless of lens and source redshifts. Fitting the profile to data requires
removing this scaling (or equivalently, scaling galaxy redshifts), which should be done through the 
cluster class interface in cluster.py

To add a profile, simply compute the projected surface mass density by integrating the 3d density
profile along the line of sight, and then compute the shear prediction by scaling delta_sigma by 
the critical surface mass density, sigma_c. For details, see Wright & Brainerd, which provides the
implementation that is used below for the NFW profile
'''

class NFW():

    def __init__(self, r200c, c, zl, cosmo=cosmo):
        '''
        This class computes the analytic tangential shear profile prediction, assuming an NFW lens

        :param r200c the radius containing mass m200c, or an average density of 200*rho_crit
        :param c: the concentration
        :param zl: the source redshift
        :param cosmo: an astropy cosmology object
        :return: None
        
        :type r200c: float
        :type c: float
        :type cosmo: object
        '''
        
        self.rs = r200c / c
        self.r200c = r200c
        self.c = c
        self.zl = zl
        self.cosmo = cosmo


    def _g(self):
        '''
        Computes the NFW prediction for the reduced shear g at the scaled radii x 
        (Eq. 14 of Wright & Brainerd 1999, with the scaling term removed)

        :return: the piecewise reduced shear g(x)
        :rettype: float array
        '''
     
        g1 = lambda x: (8.*np.arctanh(np.sqrt((1-x)/(1+x)))) / (x**2*np.sqrt(1-x**2)) +  \
                   4./x**2*np.log(x/2) - 2/(x**2-1) + \
                  (4.*np.arctanh(np.sqrt((1-x)/(1+x))) / ((x**2 - 1)*(1-x**2)**(1/2)))
        
        g2 = lambda x: (8.*np.arctan(np.sqrt((x-1)/(1+x)))) / (x**2*np.sqrt(x**2-1)) + \
                   4./x**2*np.log(x/2) - 2/(x**2-1) + \
                  (4.*np.arctan(np.sqrt((x-1)/(1+x))) / (x**2 - 1)**(3/2))
        
        # construct masks for piecewise function

        m1 = np.where(self.x < 1)
        m2 = np.where(self.x == 1)
        m3 = np.where(self.x > 1)
        
        reduced_profile = np.empty(len(self.x),dtype=np.float64)
        reduced_profile[m1] = g1( self.x[m1] )
        reduced_profile[m2] = 10./3 + 4.*np.log(1./2)
        reduced_profile[m3] = g2( self.x[m3] )
        
        return reduced_profile


    def delta_sigma(self, r):
        '''
        Computes ΔΣ for an NFW lens, given sources at projected comoving radii r. This function does not perform
        the final scaling by Σ_critical, which would result in the tangential shear γ_t = ΔΣ/Σ_c; ΔΣ, rather, 
        is the same for an identical lens, regradless of the lens or soure redshifts. Use the methods provided
        in gammaProf.cluster.lens to perform the scaling and fit to data.
        
        :param r: comoving projected radius relative to the center of the lens, per source; 
                  r = D_l*(theta^2 + phi^2)^(1/2) 
        :param zs: source redshifts
        :param zl: lens redshift
        :return: the modified surface density ΔΣ, in M_sun/pc^2
        
        :type r: array-like
        :type zs: float or array-like
        :type zl: float
        :rettype: float array
        '''

        self.x = r / self.rs

        # unit conversion factors
        cm_per_Mpc = units.Mpc.to('cm')
        kg_per_msun = const.M_sun.value

        # δ_c NFW param, 
        # critical density ρ_c in M_sun Mpc^-3,
        # modified surface density ΔΣ; rightmost factor scales Mpc to pc
        δ_c = (200/3) * self.c**3 / (np.log(1+self.c) - self.c/(1+self.c))
        ρ_c = cosmo.critical_density(self.zl).value * (cm_per_Mpc**3 / (kg_per_msun*1000))
        ΔΣ = ((self.rs*δ_c*ρ_c) * self._g()) * 1e-12

        return ΔΣ
