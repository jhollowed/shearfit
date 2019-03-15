import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7 as cosmo


class lens():
    
    def __init__(self, zl):
        """
        This class constructs a 'lens' object, which contains data vectors for the background sources
        of a given lens, and methods to perform computations on that data.

        Parameters
        ----------
        zl : float
            the redshift of the lens

        Returns
        -------
        None

        Attributes
        ----------
        zl : float
            the redshift of the lens
        has_sources : boolean
            whether or not the background population has been set for this instance
            (`False` until `set_background()` is called)
        bg_theta : float array
            the lens-centric azimuthal angular coordinate, in arcseconds
            (uninitialized until `set_background()` is called)
        bg_phi : float array
            the lens-centric coaltitude angular coordinate, in arcseconds
            (uninitialized until `set_background()` is called)
        zs : float array
            redshifts of background sources
            (uninitialized until `set_background()` is called)
        r : float array
            projected separation of each source at the redshift `zl`, in comoving Mpc
            (uninitialized until `set_background()` is called)

        Methods
        -------
        set_background(theta, phi, zs)
            Defines and assigns background souce data vectors to attributes of the lens object
        get_background()
            Returns the source population data vectors to the caller, as a list
        calc_sigma_crit()
            Computes the critical surface density at the redshift `zl`
        """

        self.zl = zl
        self.has_sources = False
        self.bg_theta = None
        self.bg_phi = None
        self.zs = None
        self.r = None


    def set_background(self, theta, phi, zs):
        '''
        Defines and assigns background souce data vectors to attributes of the lens object, 
        including the angular positions, redshifts, and projected comoving distances from 
        the lens center in Mpc.
        
        Parameters
        ----------
        theta : float array
            the lens-centric azimuthal angular coordinate, in arcseconds
        phi : float_array
            the lens-centric coaltitude angular coordinate, in arcseconds
        zs : float array
            the source redshifts
        
        Returns
        -------
        None
        '''

        self.bg_theta = (np.pi/180) * (theta/3600)
        self.bg_phi = (np.pi/180) * (phi/3600)
        self.zs = zs
        self.r = np.linalg.norm([np.tan(self.bg_theta), np.tan(self.bg_phi)], axis=0) * \
                 cosmo.comoving_distance(zs)
        self.has_sources = True


    def get_background(self):
        '''
        Returns the source population data vectors to the caller, as a list. 

        Returns
        -------
        list of numpy arrays
            A list of the source population data vectors (numpy arrays), as 
            [theta, phi, r, zs], where theta and phi are the halo-centric angular
            positions of the sources in arcseconds, r is the halo-centric 
            projected radial distance of each source in Mpc, and zs are the source
            redshifts
        '''

        return [((180/np.pi) * self.bg_theta) * 3600, 
                ((180/np.pi) * self.bg_phi) * 3600, self.r, self.zs]


    def calc_sigma_crit(self, zs=None):
        '''
        Computes :math:`\\Sigma_\\text{c}`, the critical surface density, in 
        :math:`M_{\\odot}/\\text{pc}^2`, assuming a flat cosmology

        Parameters
        ----------
        zs : float or float array, optional
            A source redshift (or array of redshifts). If None (default), then use background
            source redshifts given at object instatiation, `self.zs`
        
        Returns
        -------
        Sigma_crit : float or float array 
            The critical surface density, :math:`\\Sigma_\\text{c}`, in :math:`M_{\\odot}/\\text{pc}^2` 
        '''
        
        if(zs is None): zs = self.zs

        # unit conversions and scale factors
        m_per_mpc = units.Mpc.to('m')
        s_per_gyr = units.Gyr.to('s')
        kg_per_msun = const.M_sun.value
        a_zl = cosmo.scale_factor(self.zl)
        a_zs = cosmo.scale_factor(self.zs)

        # G in comoving Mpc^3 M_sun^-1 Gyr^-2,
        # speed of light C in comoving Mpc Gyr^-1
        # comoving distance to lens Dl and source Ds in Mpc
        # --> warning: this assumes a flat cosmology; or that angular diamter distance = proper distance
        G = const.G.value * ((s_per_gyr**2 * kg_per_msun)/m_per_mpc**3) * (1/a_zl)**3
        C = const.c.value * (s_per_gyr/m_per_mpc) * (1/a_zl)
        Ds = cosmo.angular_diameter_distance(zs).value * (1/a_zs)
        Dl = cosmo.angular_diameter_distance(self.zl).value * (1/a_zl)
        Dls = Ds - Dl
        
        # critical surface mass density Σ_c; rightmost factor scales to Mpc to pc
        Sigma_crit = (C**2/(4*np.pi*G) * (Ds)/(Dl*Dls)) * 1e-12

        return Sigma_crit 
