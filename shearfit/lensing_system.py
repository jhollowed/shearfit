import pdb
import warnings
import numpy as np
from scipy import stats
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
        The redshift of the lens.
    cosmo : object, optional
        An astropy cosmology object (defaults to `WMAP7`).

    Attributes
    ----------
    zl : float
        The redshift of the lens.
    has_sources : boolean
        Whether or not the background population has been set for this instance
        (`False` until `set_background()` is called).
    bg_theta1 : float array
        The source lens-centric azimuthal angular coordinates, in arcseconds
        (uninitialized until `set_background()` is called).
    bg_theta2 : float array
        The source lens-centric coaltitude angular coordinates, in arcseconds
        (uninitialized until `set_background()` is called).
    zs : float array
        Redshifts of background sources
        (uninitialized until `set_background()` is called).
    r : float array
        Projected separation of each source at the redshift `zl`, in comoving :math:`\\text{Mpc}`
        (uninitialized until `set_background()` is called).
    y1 : float array
        The real component of the source shears.
    y2 : float array
        The imaginary component of the source shears.
    yt : float array
        The source tangential shears.
    k : float array
        The source convergences.

    Methods
    -------
    set_background(theta1, theta2, zs, y1, y2)
        Defines and assigns background souce data vectors to attributes of the lens object.
    get_background()
        Returns the source population data vectors to the caller, as a list.
    calc_sigma_crit()
        Computes the critical surface density at the redshift `zl`.
    """
    
    def __init__(self, zl, cosmo=WMAP7):
        self.zl = zl
        self._cosmo = cosmo
        self._has_sources = False
        self._has_shear12 = False
        self._has_shear1 = False
        self._has_shear2 = False
        self._has_kappa = False
        self._has_rho = False
        self._has_radial_cuts = False
        self._rmin = None
        self._rmax = None
        self._theta1 = None
        self._theta2 = None
        self._zs = None
        self._r = None
        self._phi = None
        self._y1 = None
        self._y2 = None
        self._yt = None
        self._k =None


    def _check_sources(self):
        """
        Checks that set_background has been called (intended to be called before any
        operations on the attributes initialized by `set_background()`).
        """
        assert(self._has_sources), 'sources undefined; first run set_background()'


    def set_radial_cuts(self, rmin=None, rmax=None):
        '''
        Sets a class-wide radial mask which will be applied to data vectors returned from 
        `get_background()`, `calc_delta_sigma()`, `calc_delta_sigma_binned()`, and `calc_sigma_crit()`.

        Parameters
        ----------
        rmin : float, optional
            Sources with halo-centric radial distances less than this value will be removed by
            application of the mask constructed from this function. Defautls to None, in which case
            rmin is set to `0` (i.e. nothing is masked on the upper end of the radial distribution).
        rmax : float, optional
            Sources with halo-centric radial distances greater than this value will be removed by
            application of the mask constructed from this function. Defautls to None, in which case
            rmax is set to coincide with the maximum source radial distance (i.e. nothing is masked
            on the upper end of the radial distribution).
        '''
        self._check_sources()
        if(rmin is None): rmin = 0
        if(rmax is None): rmax = np.max(self._r)
        self._radial_mask = np.logical_and(self._r >= rmin, self._r <= rmax)
        

    def set_background(self, theta1, theta2, zs, y1=None, y2=None, yt=None, k=None, rho=None):
        '''
        Defines and assigns background souce data vectors to attributes of the lens object, 
        including the angular positions, redshifts, projected comoving distances from 
        the lens center in comoving :math:`\\text{Mpc}`, and shear components. The user 
        should either pass the shear components `y1` and `y2`, or the tangential shear `yt`; 
        if both or neither are passed, an exception will be raised.
        
        Parameters
        ----------
        theta1 : float array
            The source lens-centric azimuthal angular coordinates, in arcseconds.
        theta2 : float_array
            The source lens-centric coaltitude angular coordinates, in arcseconds.
        zs : float array
            The source redshifts.
        y1 : float array, optional
            The shear component :math:`\\gamma_1`. Must be passed along with `y2`, unless passing `yt`.
        y2 : float array, optional
            The shear component :math:`\\gamma_2`. Must be passed along with `y1`, unless passing `yt`.
        yt : float array, optional
            The tangential shear :math:`\\gamma_T`. Must be passed if not passing `y1` and `y2`.
        k : float array, optional
            The convergence :math:`\\kappa`. Not needed for any computations of this class, but is
            offered as a convenience. Defaults to `None`.
        rho : float array, optional
            The matter density at the projected source positions on the lens plane. Not needed for
            any computations of this class, but is offered as a convenience; intended use is in the
            case that the user wishes to fit `\\delta\\Sigma` directly to the projected mass density
            on the grid (output prior to ray-tracing). Defaults to `None`.
        '''

        # make sure shear was passed correctly -- either tangenetial, or components, not both
        # the _has_shear12 attribute will be used to communicate to other methods which usage
        # has been invoked
        if((y1 is None and y2 is None and yt is None) or
           ((y1 is not None or y2 is not None) and yt is not None)):
          raise Exception('Either y1 and y2 must be passed, or yt must be passed, not both.')
        
        # initialize source data vectors
        self._theta1 = np.array((np.pi/180) * (theta1/3600))
        self._theta2 = np.array((np.pi/180) * (theta2/3600))
        self._zs = np.array(zs)
        self._y1 = np.array(y1)
        self._y2 = np.array(y2)
        self._yt = np.array(yt)
        self._k = np.array(k)
        self._rho = np.array(rho)
        
        # set flags and compute additonal quantities
        if(yt is None): self._has_shear12 = True
        if(k is not None): self._has_kappa = True
        if(rho is not None): self._has_rho = True
        self._has_sources = True
        self._comp_bg_quantities()
        self.set_radial_cuts(None, None)


    def _comp_bg_quantities(self):
        """
        Computes background source quantites that depend on the data vectors initialized in 
        set_background (this function meant to be called from the setter method of each 
        source property).
        """

        self._check_sources()
        
        # compute halo-centric projected radial separation of each source, in proper Mpc
        #self._r = np.linalg.norm([np.tan(self._theta1), np.tan(self._theta2)], axis=0) * \
        #                          self._cosmo.comoving_distance(self.zl).value
        
        #arcsec_per_Mpc = (self._cosmo.arcsec_per_kpc_proper(self.zl)).to( units.arcsec / units.Mpc )
        #angular_sep_arcsec = np.linalg.norm([180/np.pi * self._theta1 * 3600, 
        #                                     180/np.pi * self._theta2 * 3600], axis=0) * units.arcsec
        #self._r = (angular_sep_arcsec / arcsec_per_Mpc).value
       
        # Projected distance in proper Mpc; Wright & Brainerd, under Eq.10
        self._r = np.linalg.norm([self._theta1, self._theta2], axis=0) * \
                                  self._cosmo.angular_diameter_distance(self.zl).value
        

        if(self._has_shear12):
            # compute tangential shear yt
            self._phi = np.arctan(self._theta2/self._theta1)
            #self._yt = -(self._y1 * np.cos(2*self._phi) + 
            #            self._y2*np.sin(2*self._phi))
            self._yt = np.sqrt(self._y1**2 + self._y2**2)
 
    
    def get_background(self):
        '''
        Returns the source population data vectors to the caller as a numpy 
        rec array, sorted in ascending order with respect to the halo-centric 
        radial distance

        Returns
        -------
        bg : 2d numpy array
            A list of the source population data vectors (2d numpy array), with
            labeled columns. 
            If shear components are being used (see docstring for `set_background()`,
            then the contents of the return array is 
            [theta1, theta2, r, zs, y1, y2, yt], where theta1 and theta2 are the 
            halo-centric angular positions of the sources in arcseconds, r is the 
            halo-centric projected radial distance of each source in proper 
            :math:`\\text{Mpc}`, zs are the source redshifts, y1 and y2 are the 
            shear components of the sources, and yt are the source tangential shears.
            If only the tangential shear is being used, then y1 and y2 are omitted
        '''
        self._check_sources()
        
        bg_arrays = [(180/np.pi * self._theta1 * 3600),
                     (180/np.pi * self._theta2 * 3600),
                     self._r, self._zs, self._yt]
        bg_dtypes = [('theta1',float), ('theta2',float), ('r',float),
                     ('zs',float), ('yt',float)]
        
        if(self._has_shear12):
            bg_arrays.append(self._y1)
            bg_arrays.append(self._y2)
            bg_dtypes.append(('y1', float))
            bg_dtypes.append(('y2', float))
        if(self._has_kappa):
            bg_arrays.append(self._k)
            bg_dtypes.append(('k', float))
        if(self._has_rho):
            bg_arrays.append(self._rho)
            bg_dtypes.append(('rho', float))
      
        bg_arrays = [arr[self._radial_mask] for arr in bg_arrays]
        bg = np.rec.fromarrays(bg_arrays, dtype = bg_dtypes) 
        return bg
     

    @property
    def cosmo(self): return self._cosmo
    @cosmo.setter
    def cosmo(self, value): 
        self._cosmo = value
        self._comp_bg_quantities()

    @property
    def r(self): return self._r
    def r(self, value):
        raise Exception('Cannot change source \'r\' value; update angular positions instead')

    @property
    def theta1(self): return self._theta1
    @theta1.setter
    def theta1(self, value): 
        self._theta1 = value
        self._comp_bg_quantities()
    
    @property
    def theta2(self): return self._theta2
    @theta2.setter
    def theta2(self, value): 
        self._theta2 = value
        self._comp_bg_quantities()
    
    @property
    def zs(self): return self._zs
    @theta1.setter
    def zs(self, value): 
        self._zs = value
        self._comp_bg_quantities()

    @property
    def k(self): return self._k
    @k.setter
    def k(self, value):
        self._k = value
        if(value is None): self._has_kappa = False
        else: self._has_kappa = True
    
    @property
    def y1(self): return self._y1
    @y1.setter
    def y1(self, value):
        if(not self._has_shear12):
            raise Exception('object initialized with yt rather than y1,y2; cannot call y1 setter')
        else:
            self._y1 = value
            self._comp_bg_quantities()
    
    @property
    def y2(self): return self._y2
    @y2.setter
    def y2(self, value): 
        if(not self._has_shear12):
            raise Exception('object initialized with yt rather than y1,y2; cannot call y2 setter')
        else:
            self._y2 = value
            self._comp_bg_quantities()
    
    @property
    def yt(self): return self._yt
    @yt.setter
    def yt(self, value): 
        self._yt = value
        if(self._has_shear12 or self._has_shear1 or self._has_shear2):
            warnings.warn('Warning: setting class attribute yt, but object was initialized' 
                          'with y1,y2 (or y1/y2 setters were called); shear components y1'
                          'and y2 being set to None')
            self._has_shear12 = False
            self._y1= None
            self._y2 = None
        self._comp_bg_quantities()
    
    @property
    def get_radial_cuts(self): return [self._rmin, self._rmax]
     

    def _k_rho(self):
        '''
        Rescales the convergence on the lens plane into a matter density. This is mostly offered
        for debugging purposes, and is only really meaningful in the case that the input is the
        raytracing of a single lens plane (otherwise the recovered matter density is cumulative, 
        in some sense, across the line of sight).
        
        Returns
        -------
        rho : float or float array 
            The projected mass density at the source positions on the lens plane
        '''
        rho = self._k * self.calc_sigma_crit()
        return rho


    def calc_sigma_crit(self, zs=None):
        '''
        Computes :math:`\\Sigma_\\text{c}(z_s)`, the critical surface density as a function of source
        redshift :math:`z_s`, at the lens redshift :math:`z_l`, in proper :math:`M_{\\odot}/\\text{pc}^2`, 
        assuming a flat cosmology

        Parameters
        ----------
        zs : float or float array, optional
            A source redshift (or array of redshifts). If None (default), then use background
            source redshifts given at object instatiation, `self.zs`
        
        Returns
        -------
        Sigma_crit : float or float array 
            The critical surface density, :math:`\\Sigma_\\text{c}`, in proper
            :math:`M_{\\odot}/\\text{pc}^2` 
        '''
        if(zs is None): 
            self._check_sources()
            zs = self._zs[self._radial_mask] 

        # G in Mpc^3 M_sun^-1 Gyr^-2,
        # speed of light C in Mpc Gyr^-1
        # distance to lens Dl and source Ds in proper Mpc
        # --> warning: this assumes a flat cosmology; or that angular diamter distance = proper distance
        G = const.G.to(units.Mpc**3 / (units.M_sun * units.Gyr**2)).value
        C = const.c.to(units.Mpc / units.Gyr).value
        Ds = self._cosmo.comoving_distance(zs).value
        Dl = self._cosmo.comoving_distance(self.zl).value
        Dls = Ds - Dl
        
        # critical surface mass density Σ_c in proper M_sun/pc^2; 
        # final quotient scales to Mpc to pc
        sigma_crit = (C**2/(4*np.pi*G) *(1.+self.zl) * (Ds)/(Dl*Dls))
        sigma_crit = sigma_crit / (1e12)

        return sigma_crit


    def calc_delta_sigma(self):
        '''
        Computes :math:`\\Delta\\Sigma = \\gamma\\Sigma_c`, the differential surface density at the lens 
        redshift :math:`z_l`, in proper :math:`M_{\\odot}/\\text{pc}^2`, assuming a flat cosmology. 
 
        Returns
        -------
        delta_sigma : float or float array 
            The differential surface density, :math:`\\Delta\\Sigma = \\gamma\\Sigma_c`, in proper
            :math:`M_{\\odot}/\\text{pc}^2 
        ''' 
        self._check_sources()
        yt = self._yt[self._radial_mask]
        sigma_crit = self.calc_sigma_crit()
        delta_sigma = yt*sigma_crit
        return delta_sigma
    
    
    def calc_delta_sigma_binned(self, nbins, return_edges=False, return_std=False, return_gradients=False):
        '''
        Computes :math:`\\Delta\\Sigma = \\gamma\\Sigma_c`, the differential surface density at the lens 
        redshift :math:`z_l`, in proper :math:`M_{\\odot}/\\text{pc}^2`, assuming a flat cosmology. 
 
        Parameters
        ----------
        nbins : int
            Number of bins to place the data into. The bin edges will be distributed uniformly in radial space
            (i.e. the bin widths will be constant, rather than bin areas)
        return_edges : bool, optional
            whether or not to return the resulting bin edges. Defautls to False
        return_std : bool, optional
            Whether or not to return the standard deviation and standard error of the mean of each bin. 
            Defaults to False.
        return_gradients : bool, optional
            Whether or not to return the approximate gradient of each bin. The gradient is computed by
            fitting a linear form to each bin's data, and returning the slope parameter. Defaults to False.

        Returns
        -------
        delta_sigma : float or float array 
            The differential surface density, :math:`\\Delta\\Sigma = \\gamma\\Sigma_c`, in proper
            :math:`M_{\\odot}/\\text{pc}^2 
        '''
        
        self._check_sources()
       
        # load data and sort by increasing radial distance
        r = self._r[self._radial_mask]
        sorter = np.argsort(r)
        r = r[sorter]
        
        yt = self._yt[self._radial_mask][sorter]
        sigma_crit = self.calc_sigma_crit()[sorter]
        delta_sigma = yt*sigma_crit
        
        # get bin means
        [r_mean, bin_edges, _] = stats.binned_statistic(r, r, statistic='mean', bins=nbins)
        [delta_sigma_mean,_,_] = stats.binned_statistic(r, delta_sigma, statistic='mean', bins=nbins)
        return_arrays = [r_mean, delta_sigma_mean] 
        return_cols = ['r_mean', 'delta_sigma_mean']
       
       # and standard deviations, errors of the mean
        if(return_std):
            [delta_sigma_std,_,_] = stats.binned_statistic(r, delta_sigma, statistic='std', bins=nbins)
            [delta_sigma_count,_,_] = stats.binned_statistic(r, delta_sigma, statistic='count', bins=nbins)
            delta_sigma_se = delta_sigma_std / delta_sigma_count
            
            [r_std,_,_] = stats.binned_statistic(r, r, statistic='std', bins=nbins)
            [r_count,_,_] = stats.binned_statistic(r, r, statistic='count', bins=nbins)
            r_se = r_std / r_count
            
            return_arrays.extend([r_std, r_se, delta_sigma_std, delta_sigma_se]) 
            return_cols.extend(['r_std', 'r_se_mean', 'delta_sigma_std', 'delta_sigma_se_mean']) 

        # return bin edges
        if(return_edges):
            return_arrays.append(bin_edges)
            return_cols.append('bin_edges')
        
        # return bin gradient and errors... compute these manually
        if(return_gradients): 
            bin_gradients  = np.zeros(nbins)
            for i in range(nbins):
                
                bin_mask = np.logical_and(r > bin_edges[i], r < bin_edges[i+1])
                if(np.sum(bin_mask) == 0): bin_gradients[i] = float('NaN')
                else:
                    ds, dr = delta_sigma[bin_mask], r[bin_mask]
                    bin_gradients[i],_ = np.polyfit(dr, ds, 1)    
            
            return_arrays.append(bin_gradients)
            return_cols.append('bin_grad')
        
        # gather for return
        bin_dict = {}
        for i in range(len(return_arrays)): bin_dict[return_cols[i]] = return_arrays[i]
        return bin_dict
