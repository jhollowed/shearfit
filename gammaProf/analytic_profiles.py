import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7
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

class NFW:
    """
    This class computes the analytic tangential shear profile prediction, assuming an NFW lens
    
    Parameters
    ----------
    r200c : float
        the radius containing mass :math:`M_{200c}`, or an average density of 
        :math:`200\\rho_\\text{crit}`
    c : float 
        the concentration
    zl : float 
        the source redshift
    cosmo : object, optional
        an astropy cosmology object (defaults to WMAP7)
    
    Attributes
    ----------
    rs : float
        the scale radius, `r200c / c`
    x : float array
        the normalized radii `r/rs` for any input `r` 
        (not initialized until `delta_sigma()` is called) 

    Methods
    -------
    delta_sigma(r)
        Computes :math:`\\Delta\\Sigma(r)` for an NFW lens, given sources at projected 
        comoving radii :math:`r`
    get_params()
        Returns the NFW radius and concentration parameters to the caller
    set_params(r200c, c)
        Returns the NFW radius and concentration parameters to the caller
    """
    def __init__(self, r200c, c, zl, cosmo=WMAP7): 
        self._r200c = r200c
        self._c = c
        self.zl = zl
        self.cosmo = cosmo
        self._rs = r200c / c
        self._x = None

    @property
    def r200c(self): return self._r200c
    @r200c.setter
    def r200c(self, value):
        assert(isinstance(value, float))
        self._r200c = value
        self._rs = self.r200c / self.c
    
    @property
    def c(self): return self._c
    @c.setter
    def c(self, value): 
        assert(isinstance(value, float))
        self._c = value
        self._rs = self.r200c / self.c


    def radius_to_mass(self):
        """
        Computes the halo mass contained within :math:`r_{200c}`

        Returns
        -------
        m200c : float
            The halo mass :math:`M_{200c}` in units of :math:`M_\\odot`
        """
       
        # critical density in M_sun / Mpc^3
        cm_per_Mpc = units.Mpc.to('cm')
        kg_per_msun = const.M_sun.value
        rho_crit = self.cosmo.critical_density(self.zl).value * (cm_per_Mpc**3 / (kg_per_msun*1000))

        m200c = (4/3) * np.pi * self._r200c**3 * rho_crit
        return m200c


    def _g(self, x):
        """
        Computes the NFW prediction for the reduced shear g at the scaled radii x 
        (Eq. 14 of Wright & Brainerd 1999, with the scaling term removed)
        
        Parameters
        ----------
        x : float array
            the normalized radii r/r_s at which to compute the reduced shear g(x)

        Returns
        -------
        reduced_profile : float array
            the piecewise reduced shear :math:`g(x)`
        """
     
        g1 = lambda x: (8.*np.arctanh(np.sqrt((1-x)/(1+x)))) / (x**2*np.sqrt(1-x**2)) +  \
                   4./x**2*np.log(x/2) - 2/(x**2-1) + \
                  (4.*np.arctanh(np.sqrt((1-x)/(1+x))) / ((x**2 - 1)*(1-x**2)**(1/2)))
        
        g2 = lambda x: (8.*np.arctan(np.sqrt((x-1)/(1+x)))) / (x**2*np.sqrt(x**2-1)) + \
                   4./x**2*np.log(x/2) - 2/(x**2-1) + \
                  (4.*np.arctan(np.sqrt((x-1)/(1+x))) / (x**2 - 1)**(3/2))
        
        # construct masks for piecewise function
        m1 = np.where(x < 1)
        m2 = np.where(x == 1)
        m3 = np.where(x > 1)
        
        reduced_profile = np.empty(len(x),dtype=np.float64)
        reduced_profile[m1] = g1( x[m1] )
        reduced_profile[m2] = 10./3 + 4.*np.log(1./2)
        reduced_profile[m3] = g2( x[m3] )
        
        return reduced_profile


    def delta_sigma(self, r):
        """
        Computes :math:`\\Delta\\Sigma` at projected comoving radii `r`, for an NFW lens. 
        This function does not perform the final scaling by :math:`\\Sigma_{\\text{critical}}`, 
        which would result in the tangential shear :math:`\\gamma_T = \\Delta\\Sigma/\\Sigma_\\text{c}`; 
        :math:`\\Delta\\Sigma`, rather, is the same for an identical lens, regradless of the lens or 
        soure redshifts. Use the methods provided in `gammaProf.cluster.lens` to perform the scaling and 
        fit to data.
        
        Parameters
        ----------
        r : float array
            comoving projected radius relative to the center of the lens; 
            :math:`r = D_l\\sqrt{\\theta_1^2 + \\theta_2^2}`
        
        Returns
        -------
        dSigma : float array
            the modified surface density :math:`\\Delta\\Sigma` in :math:`M_{\\odot}/\\text{pc}^2` 
        """

        x = r / self._rs

        # unit conversion factors
        cm_per_Mpc = units.Mpc.to('cm')
        kg_per_msun = const.M_sun.value
        pc_per_Mpc = 1e12

        # del_c NFW param, 
        # critical density rho_crit in M_sun Mpc^-3,
        # modified surface density DSigma; rightmost factor scales Mpc to pc
        del_c = (200/3) * self._c**3 / (np.log(1+self._c) - self.c/(1+self._c))
        rho_crit = self.cosmo.critical_density(self.zl).value * (cm_per_Mpc**3 / (kg_per_msun*1000))
        dSigma = ((self._rs*del_c*rho_crit) * self._g(x)) / pc_per_Mpc

        return dSigma
