import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7 as cosmo

class obs_lens_system:
    """
    This class constructs an object representing an observer-lens-source system, which contains the 
    lens redshift, and data vectors for the background sources of a given lens, including the lensing
    geometry, and methods to perform computations on that data.

    Parameters
    ----------
    zl : float
        the redshift of the lens

    Attributes
    ----------
    zl : float
        the redshift of the lens
    has_sources : boolean
        whether or not the background population has been set for this instance
        (`False` until `set_background()` is called)
    bg_theta1 : float array
        the source lens-centric azimuthal angular coordinates, in arcseconds
        (uninitialized until `set_background()` is called)
    bg_theta2 : float array
        the source lens-centric coaltitude angular coordinates, in arcseconds
        (uninitialized until `set_background()` is called)
    zs : float array
        redshifts of background sources
        (uninitialized until `set_background()` is called)
    r : float array
        projected separation of each source at the redshift `zl`, in comoving Mpc
        (uninitialized until `set_background()` is called)
    y1 : float array
        the real component of the source shears
    y2 : float array
        the imaginary component of the source shears
    yt : float array
        the source tangential shears

    Methods
    -------
    set_background(theta1, theta2, zs, y1, y2)
        Defines and assigns background souce data vectors to attributes of the lens object
    get_background()
        Returns the source population data vectors to the caller, as a list
    calc_sigma_crit()
        Computes the critical surface density at the redshift `zl`
    """
    
    def __init__(self, zl):
        self.zl = zl
        self.has_sources = False
        self.bg_theta1 = None
        self.bg_theta2 = None
        self.zs = None
        self.r = None
        self.y1 = None
        self.y2 = None
        self.yt = None


    def set_background(self, theta1, theta2, zs, y1, y2):
        '''
        Defines and assigns background souce data vectors to attributes of the lens object, 
        including the angular positions, redshifts, projected comoving distances from 
        the lens center in Mpc, and shear components.
        
        Parameters
        ----------
        theta1 : float array
            the source lens-centric azimuthal angular coordinates, in arcseconds
        theta2 : float_array
            the source lens-centric coaltitude angular coordinates, in arcseconds
        zs : float array
            the source redshifts
        y1 : float array
            the shear component :math:`\\gamma_1`
        y2 : float array
            the shear component :math:`\\gamma_2`
        
        Returns
        -------
        None
        '''

        self.bg_theta1 = (np.pi/180) * (theta1/3600)
        self.bg_theta2 = (np.pi/180) * (theta2/3600)
        self.zs = zs
        self.r = np.linalg.norm([np.tan(self.bg_theta1), np.tan(self.bg_theta2)], axis=0) * \
                 cosmo.comoving_distance(zs)
        
        # compute tangential shear yt
        phi = np.arctan(theta2/theta1)
        self.y1 = y1
        self.y2 = y2
        self.yt = -(y1 * np.cos(2*phi) + y2*np.sin(2*phi))

        self.has_sources = True


    def get_background(self):
        '''
        Returns the source population data vectors to the caller, as a list. 

        Returns
        -------
        list of numpy arrays
            A list of the source population data vectors (numpy arrays), as 
            [theta1, theta2, r, zs, y1, y2, yt], where theta1 and theta2 are the 
            halo-centric angular positions of the sources in arcseconds, r is the 
            halo-centric projected radial distance of each source in Mpc, zs are 
            the source redshifts, y1 and y2 are the shear components of the sources, 
            and yt are the source tangential shears
        '''

        return [((180/np.pi) * self.bg_theta1) * 3600, 
                ((180/np.pi) * self.bg_theta2) * 3600, 
                self.r, self.zs, selfy1, self.y2, self.yt]


    def calc_sigma_crit(self, zs=None):
        '''
        Computes :math:`\\Sigma_\\text{c}(z_s)`, the critical surface density as a function of source
        redshift :math:`z_s`, at the lens redshift :math:`z_l`, in :math:`M_{\\odot}/\\text{pc}^2`, 
        assuming a flat cosmology

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
        pc_per_Mpc = 1e12
        a_zl = cosmo.scale_factor(self.zl)

        # G in comoving Mpc^3 M_sun^-1 Gyr^-2,
        # speed of light C in comoving Mpc Gyr^-1
        # comoving distance to lens Dl and source Ds in Mpc
        # --> warning: this assumes a flat cosmology; or that angular diamter distance = proper distance
        G = const.G.value * ((s_per_gyr**2 * kg_per_msun)/m_per_mpc**3)
        C = const.c.value * (s_per_gyr/m_per_mpc)
        Ds = cosmo.angular_diameter_distance(zs).value
        Dl = cosmo.angular_diameter_distance(self.zl).value
        Dls = Ds - Dl
        
        # critical surface mass density Σ_c; 
        # rightmost product scales to Mpc to pc, and then to a comoving surface area
        Sigma_crit = (C**2/(4*np.pi*G) * (Ds)/(Dl*Dls)) / (pc_per_Mpc * a_zl**2)

        return Sigma_crit 
