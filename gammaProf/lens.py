import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7 as cosmo

class lens():
    
    def __init__(self, zl):
        '''
        This class constructs a 'lens' object, which contains data vectors for the background sources
        of a given lens, and methods to perform computations on that data.

        :param zl: the redshift of the lens
        :return: None
        :type zl: float
        '''

        self.zl = zl
        self.has_sources = False


    def set_background(self, theta, phi, zs):
        '''
        Defines and assigns background souce data vectors to attributes of the lens object, 
        including the angular positions, redshifts, and projected comoving distances from 
        the lens center.

        :param theta: the lens-centric azimuthal angular coordinate, in arcseconds
        :param phi: the lens-centric coaltitude angular coordinate, in arcseconds
        :zs: the source redshifts
        :return: None
        :type theta: float array
        :type phi: float array
        :type zs: float array
        '''

        self.bg_theta = (np.pi/180) * (theta/3600)
        self.bg_phi = (np.pi/180) * (phi/3600)
        self.zs = zs
        self.r = np.linalg.norm([np.tan(self.bg_theta), np.tan(self.bg_phi)], axis=0) * \
                 cosmo.comoving_distance(zs)
        self.has_sources = True


    def get_background(self):
        return [self.bg_theta, self.bg_phi, self.zs]


    def calc_sigma_crit(self, zs=None):
        '''
        Computes Σ_c, the critical surface density, in M_sun/pc^2, assuming a flat cosmology

        :param zs: optionally, a source redshift (array). If None (default), then use background
                   source redshifts given at object instatiation
        :return: Σ_c, the critical surface density, in M_sun/pc^2
        
        :type zs: float, float array, or None
        :rettype: float or float array
        '''
        
        if(zs is None): zs = self.zs

        # unit conversion factors
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
        Σ_c = (C**2/(4*np.pi*G) * (Ds)/(Dl*Dls)) * 1e-12

        return Σ_c 
