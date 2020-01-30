import pdb
import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7

'''
This module contains a collection of ananlytic halo profile forms (currently just NFW), which 
return a prediction for the induced source tangential shear by the lens, as a function of halocentric 
projected radius. Specifically, the scaled tangential shear (critical surface density mutliplied through 
the shearis returned, which is a constant for any identical lens, regardless of lens and source redshifts.
Fitting the profile to data requires removing this scaling (or equivalently, scaling galaxy redshifts), 
which should be done through the cluster class interface in `lensing_system.py`.

To add a profile, simply compute the projected surface mass density by integrating the 3d density
profile along the line of sight, and then compute the shear prediction by scaling delta_sigma by 
the critical surface mass density, sigma_c. For details, see Wright & Brainerd, which provides the
implementation that is used below for the NFW profile.
'''

class NFW:
    """
    This class computes the analytic tangential shear profile prediction, assuming an NFW lens.
    
    Parameters
    ----------
    r200c : float
        The comoving radius containing mass :math:`M_{200c}`, or an average density of 
        :math:`200\\rho_\\text{crit}`, in :math:`Mpc/h`.
    c : float 
        The concentration.
    zl : float 
        The source redshift.
    r200c_err : float, optional
        The 1-sigma error in the radius, in :math:`Mpc/h`. Defaults to 0.
    c_err : float, optional
        The 1-sigma error in the concentration. Defaults to 0.
    cosmo : object, optional
        An AstroPy `cosmology` object. Defaults to `WMAP7`.
    
    Attributes
    ----------
    rs : float
        The scale radius, `r200c / c` in comoving :math:`Mpc/h`.
    x : float array
        The dimensionless radii `r/rs` for any input `r` 
        (not initialized until `delta_sigma()` is called).

    Methods
    -------
    delta_sigma(r)
        Computes :math:`\\Delta\\Sigma(r)` for an NFW lens, given sources at projected 
        comoving radii :math:`r`.
    radius_to_mass():
        Converts the :math:`r_{200c}` radius of the halo to a mass in :math:`M_\\odot/h`
    """

    def __init__(self, r200c, c, zl, r200c_err=0, c_err=0, cosmo=WMAP7): 
        
        self.zl = zl
        self._cosmo = cosmo
        
        self._c = c
        self.c_err = c_err
        self._del_c = (200/3) * self._c**3 / (np.log(1+self._c) - self.c/(1+self._c))
        
        self._r200c = r200c
        self.r200c_err = r200c_err
        self._rs = r200c / c
        self._x = None

    @property
    def r200c(self): return self._r200c
    @r200c.setter
    def r200c(self, value):
        assert(isinstance(value, float))
        self._r200c = value
        self.update_params()
    
    @property
    def rs(self): return self._rs
    
    @property
    def c(self): return self._c
    @c.setter
    def c(self, value): 
        assert(isinstance(value, float))
        self._c = value
        self.update_params()

    @property
    def del_c(self): return self._del_c


    def update_params(self):
        """
        Recompute parameters that are sensitive to changes in r200c or c, in the event that they are
        modified externally.
        """
        self._rs = self.r200c / self.c
        self._del_c = (200/3) * self._c**3 / (np.log(1+self._c) - self.c/(1+self._c))
 

    def radius_to_mass(self):
        """
        Computes the halo mass contained within :math:`r_{200c}`.

        Returns
        -------
        m200c : float
            The halo mass :math:`M_{200c}` in units of :math:`M_\\odot/h`.
        """
       
        # critical density in proper (M_sun/h) / (Mpc/h)^3
        rho_crit = self._cosmo.critical_density(self.zl)
        rho_crit = rho_crit.to(units.Msun/units.Mpc**3).value / self._cosmo.h**2
        a = 1/(1+self.zl)

        m200c = (4/3) * np.pi * (self._r200c * a)**3 * (rho_crit * 200)
        return m200c


    def _g(self, x):
        """
        Computes the NFW prediction for the reduced shear g at the scaled radii x 
        (Eq. 14 of Wright & Brainerd 1999, with the scaling term removed).
        
        Parameters
        ----------
        x : float array
            The normalized radii r/r_s at which to compute the reduced shear g(x).

        Returns
        -------
        reduced_profile : float array
            The piecewise reduced shear :math:`g(x)`.
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


    def delta_sigma(self, r, bootstrap=False, bootN=1000):
        """
        Computes :math:`\\Delta\\Sigma` at projected comoving radii `r`, for an NFW lens. 
        This function does not perform the final scaling by :math:`\\Sigma_{\\text{critical}}`, 
        which would result in the tangential shear :math:`\\gamma_T = \\Delta\\Sigma/\\Sigma_\\text{c}`; 
        :math:`\\Delta\\Sigma`, rather, is the same for an identical lens, regradless of the lens or 
        soure redshifts. Use the methods provided in `gammaProf.lensing_system` to perform the scaling and 
        fit to data.
        
        Parameters
        ----------
        r : float array
            Comoving projected radius relative to the center of the lens, in :math:`Mpc/h`; 
            :math:`r = D_l\\sqrt{\\theta_1^2 + \\theta_2^2}`.
        bootstrap : boolean, optional
            Whether or not to perform a bootstrap resampling of the :math:`r_{200c}` and :math:`c`
            parameters, to estimate the error on :math:`\\Delta\\Sigma(r)`. If `True`, then compute
            the projected surface density `bootN` times, drawing `r200c` and `c` from a Gaussian 
            distribution :math:`N(\\mu, \\sigma**2)`, where :math:`\\sigma` is given by `r200c_err` 
            and `c_err`. Defaults to `False`.
        bootN : int, optional
            Number of bootstrap resamples to perform. Defaults to `1000`.
        
        Returns
        -------
        float array or list of float arrays
            The modified surface density :math:`\\Delta\\Sigma` in comoving :math:`(M_{\\odot}/h)/(\\text{pc}/h)^2`, 
            for each value of `r`. If `bootstrap` is `True`, then pack this array into a two-element 
            list, which is followed by the estaimted :math:`1\\sigma` error at each of those locations.
        """
        
        if(bootstrap == False):
            return self._delta_sigma(r)
        
        else:
            assert self.r200c_err != 0 and self.c_err != 0, "bootstrap option should be "\
                   "disabled if radius or concentration errors are zero"
            
            # prep bootstrap and save current radius and concentration to restore after resampling
            r200c = self.r200c
            c = self.c
            dsig_bootstrap = np.zeros((len(r), bootN))
            dsig_stderr = np.zeros((len(r), 2))
            
            # do resampling of r200c and c
            r200c_resamples = self.r200c_err * np.random.randn(bootN) + self.r200c
            c_resamples = self.c_err * np.random.randn(bootN) + self.c

            # calculate projected surface density for each realization
            for i in range(bootN):
                self.r200c = r200c_resamples[i]
                self.c = c_resamples[i]
                dsig_bootstrap[i] = self._delta_sigma(r)

            # estimate asymmetric 1-sigma errors
            for i in range(len(r)):
                dsig_r = np.sort(dsig_bootstrap.T[i])
                dsig_r = dsig_r[~np.isnan(dsig_r)]
                u = np.mean(dsig_r)
                try:
                    down1sig = dsig_r[ np.searchsorted(dsig_r, u) - int(len(dsig_r)*0.341) ]
                    up1sig = dsig_r[ np.searchsorted(dsig_r, u) + int(len(dsig_r)*0.341) ]
                    dsig_stderr[i][:] = [u-down1sig, up1sig-u]
                except IndexError:
                    dsig_stderr[i][:] = dsig_stderr[i-1][:]

            # restore profile and compute delta sigma; return
            self.r200c = r200c
            self.c = c
            return [self._delta_sigma(r), dsig_stderr]


    def _delta_sigma(self, r):
        """
        Computes :math:`\\Delta\\Sigma` at projected comoving radii `r`, for an NFW lens. The implementation
        is Eq.(14) given in Wright & Brainerd 1999, with the critical density factor removed. Hence, 
        this function does not perform the final scaling by :math:`\\Sigma_{\\text{critical}}`, 
        which would result in the tangential shear :math:`\\gamma_T = \\Delta\\Sigma/\\Sigma_\\text{c}`; 
        :math:`\\Delta\\Sigma`, rather, is the same for an identical lens, regradless of the lens or 
        soure redshifts. Use the methods provided in `gammaProf.cluster.lens` to perform the scaling and 
        fit to data.
        
        Parameters
        ----------
        r : float array
            Comoving projected radius relative to the center of the lens; 
            :math:`r = D_l\\sqrt{\\theta_1^2 + \\theta_2^2}`, in comoving :math:`Mpc/h`.
        
        Returns
        -------
        dSigma : float array
            The modified surface density :math:`\\Delta\\Sigma` in comoving :math:`(M_{\\odot}/h)/(\\text{pc}/h)^2`
        """

        # 1e6 in rs to get Mpc to pc
        rs = self._rs * 1e6
        x = r / self._rs
        a = 1/(1+self.zl)

        # define critical density rho_crit in proper M_sun pc^-3,
        # 1/h^2 in rho_crit to get units to match radius
        rho_crit = self._cosmo.critical_density(self.zl)
        rho_crit = rho_crit.to(units.Msun/units.pc**3).value / self._cosmo.h**2
        
        # proper mean surface density DSigma in Mpc/h
        # factor of a^2 to get comoving surface area
        dSigma = (rs * self._del_c * rho_crit) * self._g(x)
        dSigma = dSigma * a**2

        return dSigma


    def sigma(self, r):
        """
        Computes :math:`\\Sigma` at projected comoving radii `r`, for an NFW lens. The implementation
        is Eq.(11) given in Wright & Brainerd 1999. 
        
        Parameters
        ----------
        r : float array
            Comoving projected radius relative to the center of the lens; 
            :math:`r = D_l\\sqrt{\\theta_1^2 + \\theta_2^2}`, in comoving :math:`Mpc/h`.
        
        Returns
        -------
        dSigma : float array
            The modified surface density :math:`\\Delta\\Sigma` in comoving :math:`(M_{\\odot}/h)/(\\text{pc}/h)^2`
        """
 
        # 1e6 in rs to get Mpc to pc
        rs = self._rs * 1e6
        x = r / self._rs
        a = 1/(1+self.zl)

        # define critical density rho_crit in proper M_sun pc^-3,
        # 1/h^2 in rho_crit to get units to match radius
        rho_crit = self._cosmo.critical_density(self.zl)
        rho_crit = rho_crit.to(units.Msun/units.pc**3).value / self._cosmo.h**2
         
        # NFW prediction for surface density
        f1 = lambda x: (2/(x**2-1)) * (1 - ( 2/np.sqrt(1-x**2) * np.arctanh(np.sqrt((1-x)/(1+x))) ))
        f2 = lambda x: (2/(x**2-1)) * (1 - ( 2/np.sqrt(x**2-1) * np.arctan(np.sqrt((x-1)/(1+x))) ))
        
        # construct masks for piecewise function
        m1 = np.where(x < 1)
        m2 = np.where(x == 1)
        m3 = np.where(x > 1)
        
        sigma = np.empty(len(x),dtype=np.float64)
        sigma[m1] = f1( x[m1] )
        sigma[m2] = 2./3.
        sigma[m3] = f2( x[m3] )

        # add cosmology dependence prefactor, and a^2 to get comoving surface area
        prefactor = rs * self._del_c * rho_crit
        sigma_nfw = prefactor * sigma * a**2
        
        return sigma_nfw
