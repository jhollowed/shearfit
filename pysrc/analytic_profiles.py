import numpy as np
import astropy.units as units
import astropy.constants as const
from astropy.cosmology import WMAP7 as cosmo
import pdb

'''
This module contains a collection of ananlytic halo profile forms (currently just NFW), which 
return a prediction for the average tangential shear as a function of halocentric projected radius

To add a profile, simply compute the projected surface mass density by integrating the 3d density
profile along th line of sight, and then compute the shear prediction by scaling delta_sigma by 
the critical surface mass density, sigma_c. For details, see Wright & Brainerd, which provides the
implementation that is used below for the NFW profile
'''

class NFW_shear_profile():

    def __init__(self, r200c, c, cosmo=cosmo):
        '''
        :param r200c the radius containing mass m200c, or an average density of 200*rho_crit
        :param c: the concentration
        :param cosmo: an astropy cosmology object

        :type r200c: float
        :type c: float
        :type cosmo: object
        '''
        
        self.rs = r200c / c
        self.r200c = r200c
        self.c = c
        self.cosmo = cosmo


    def _g(self):
        '''
        Computes the NFW prediction for the reduced shear g at the scaled radii x 
        (Eq. 14 of Wright & Brainerd 1999, with the scaling term removed)
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


    def scaled_prediction(self, r, zs, zl):
        '''
        Computes the predicted tangential shear profile for an NFW lens, given sources at projected radii r, 
        and redshifts zs.
        
        :param r: projected radius relative to the center of the lens, per source; r = D_l*(theta^2 + phi^2)^(1/2)
        :param zs: source redshifts
        :param zl: lens redshift
        
        :type r: array-like
        :type zs: float or array-like
        :type zl: float
        '''

        self.x = r / self.rs

        # m and cm per Mpc
        m_per_mpc = units.Mpc.to('m')
        cm_per_mpc = units.Mpc.to('cm')
        s_per_gyr = units.Gyr.to('s')
        kg_per_msun = const.M_sun.value

        # delta-c NFW param, 
        # critical density pc in M_sun Mpc^-3, 
        # G in Mpc^3 M_sun^-1 Gyr^-2, 
        # speed of light C in Mpc Gyr^-1
        # critical surface mass density Sc
        # angular diameter distance to lens Dl and source Ds in Mpc
        dc = (200/3) * self.c**3 / (np.log(1+self.c) - self.c/(1+self.c))
        pc = cosmo.critical_density(zl).value * (cm_per_mpc**3 / (kg_per_msun*1000))
        G = const.G.value * ((s_per_gyr**2 * kg_per_msun)/m_per_mpc**3)
        C = const.c.value * (s_per_gyr/m_per_mpc)
        Ds = cosmo.angular_diameter_distance(zs).value
        Dl = cosmo.angular_diameter_distance(zl).value
        Dls = Ds - Dl
        Sc = C**2/(4*np.pi*G) * (Ds)/(Dl*Dls)
        
        profile = (self.rs*dc*pc) / Sc * self._g()
        return profile
