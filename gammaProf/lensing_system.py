import warnings
import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7

class obs_lens_system:
    """
    This class constructs an object representing an observer-lens-source system, which contains the 
    lens redshift, and data vectors for the background sources of a given lens, including the lensing
    geometry, and methods to perform computations on that data.

    Parameters
    ----------
    zl : float
        the redshift of the lens
    cosmo : object, optional
        an astropy cosmology object (defaults to WMAP7)

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
    
    def __init__(self, zl, cosmo=WMAP7):
        self.zl = zl
        self.cosmo = cosmo
        self._has_sources = False
        self._has_shear12 = False
        self.bg_theta1 = None
        self.bg_theta2 = None
        self.zs = None
        self._r = None
        self._phi = None
        self.y1 = None
        self.y2 = None
        self.yt = None


    def _check_sources(self):
        """
        Checks that set_background has been called (intended to be called before any
        operations on the attributes initialized by set_background())
        """
        assert(self._has_sources), 'sources undefined; first run set_background()'
        

    def set_background(self, theta1, theta2, zs, y1=None, y2=None, yt=None):
        '''
        Defines and assigns background souce data vectors to attributes of the lens object, 
        including the angular positions, redshifts, projected comoving distances from 
        the lens center in Mpc, and shear components. The user should either pass the shear 
        components `y1` and `y2`, or the tangential shear `yt`; if both or neither are passed, 
        an exception will be raised
        
        Parameters
        ----------
        theta1 : float array
            the source lens-centric azimuthal angular coordinates, in arcseconds
        theta2 : float_array
            the source lens-centric coaltitude angular coordinates, in arcseconds
        zs : float array
            the source redshifts
        y1 : float array, optional
            the shear component :math:`\\gamma_1`
        y2 : float array, optional
            the shear component :math:`\\gamma_2`
        yt : float array, optional
            the tangential shear :math:`\\gamma_T`
        
        Returns
        -------
        None
        '''

        # make sure shear was passed correctly -- either tangenetial, or components, not both
        # the _has_shear12 attribute will be used to communicate to other methods which usage
        # has been invoked
        if((y1 is None and y2 is None and yt is None) or
           ((y1 is not None or y2 is not None) and yt is not None):
          raise Exception('Either y1 and y2 must be passed, or yt must be passed, not both.')
        
        # initialize source data vectors
        self.bg_theta1 = (np.pi/180) * (theta1/3600)
        self.bg_theta2 = (np.pi/180) * (theta2/3600)
        self.zs = zs
        self.y1 = y1
        self.y2 = y2
        self.yt = yt
        
        # set flags and compute additonal quantities
        if(yt is None): self._has_shear12 = True
        self._has_sources = True
        self._comp_bg_quantities()


    def _comp_bg_quantities(self):
        """
        Computes background source quantites that depend on the data vectors initialized in 
        set_baclground (this function meant to be called from the setter method of each 
        source property)
        """

        self._check_sources()
        
        # compute halo-centric projected radial separation of each source, in Mpc
        self._r = np.linalg.norm([np.tan(self.bg_theta1), np.tan(self.bg_theta2)], axis=0) * \
                                  self.cosmo.comoving_distance(zs)
        
        if(self._has_shear12):
            # compute tangential shear yt
            self._phi = np.arctan(theta2/theta1)
            self.yt = -(y1 * np.cos(2*phi) + y2*np.sin(2*phi))
    
    
    def get_background(self):
        '''
        Returns the source population data vectors to the caller, as a list. 

        Returns
        -------
        bg : 2d numpy array
            A list of the source population data vectors (2d numpy array), with
            labeled columns. 
            If shear components are being used (see docstring for `set_background()`,
            then the contents of the return array is 
            [theta1, theta2, r, zs, y1, y2, yt], where theta1 and theta2 are the 
            halo-centric angular positions of the sources in arcseconds, r is the 
            halo-centric projected radial distance of each source in Mpc, zs are 
            the source redshifts, y1 and y2 are the shear components of the sources, 
            and yt are the source tangential shears.
            If only the tangential shear is being used, then y1 and y2 are omitted
        '''
        
        self._check_sources()
        
        if(self._has_shear12):
            bg = np.array([((180/np.pi) * self.bg_theta1) * 3600, 
                           ((180/np.pi) * self.bg_theta2) * 3600, 
                           self._r, self.zs, self.y1, self.y2, self.yt], 
                           dtype = [('theta1',float), ('theta2',float), 
                                    ('r',float), ('zs',float), 
                                    ('y1',float), ('y2',float), ('yt',float)])
        else:
            bg = np.array([((180/np.pi) * self.bg_theta1) * 3600, 
                           ((180/np.pi) * self.bg_theta2) * 3600, 
                           self._r, self.zs, self.yt], 
                           dtype = [('theta1',float), ('theta2',float), 
                                    ('r',float), ('zs',float), ('yt',float)])
        return bg
    

    @property
    def cosmo(self): return self.cosmo
    @cosmo.setter
    def cosmo(self, value): 
        self.cosmo = value
        self._comp_bg_quantities()

    @property
    def theta1(self): return self.theta1
    @theta1.setter
    def theta1(self, value): 
        self.theta1 = value
        self._comp_bg_quantities()
    
    @property
    def theta2(self): return self.theta2
    @theta2.setter
    def theta2(self, value): 
        self.theta2 = value
        self._comp_bg_quantities()
    
    @property
    def zs(self): return self.zs
    @theta1.setter
    def zs(self, value): 
        self.zs = value
        self._comp_bg_quantities()
    
    @property
    def y1(self): return self.y1
    @y1.setter
    def y1(self, value): 
        self.y1 = value
        self._comp_bg_quantities()
    
    @property
    def y2(self): return self.y2
    @y2.setter
    def y2(self, value): 
        self.y2 = value
        self._comp_bg_quantities()
    
    @property
    def yt(self): return self.yt
    @yt.setter
    def yt(self, value): 
        self.yt = value
        if(self._has_shear12):
            warnings.warn('Warning: setting class attribute yt, but object was initialized 
                           with y1,y2; shear components y1 and y2 being set to None')
            self._has_shear12 = False
            self.y1= None
            self.y2 = None
        self._comp_bg_quantities()


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
        
        self._check_sources()
        if(zs is None): zs = self.zs

        # unit conversions and scale factors
        m_per_mpc = units.Mpc.to('m')
        s_per_gyr = units.Gyr.to('s')
        kg_per_msun = const.M_sun.value
        pc_per_Mpc = 1e12
        a_zl = self.cosmo.scale_factor(self.zl)

        # G in comoving Mpc^3 M_sun^-1 Gyr^-2,
        # speed of light C in comoving Mpc Gyr^-1
        # comoving distance to lens Dl and source Ds in Mpc
        # --> warning: this assumes a flat cosmology; or that angular diamter distance = proper distance
        G = const.G.value * ((s_per_gyr**2 * kg_per_msun)/m_per_mpc**3)
        C = const.c.value * (s_per_gyr/m_per_mpc)
        Ds = self.cosmo.angular_diameter_distance(zs).value
        Dl = self.cosmo.angular_diameter_distance(self.zl).value
        Dls = Ds - Dl
        
        # critical surface mass density Σ_c; 
        # rightmost product scales to Mpc to pc, and then to a comoving surface area
        Sigma_crit = (C**2/(4*np.pi*G) * (Ds)/(Dl*Dls)) / (pc_per_Mpc * a_zl**2)

        return Sigma_crit 
