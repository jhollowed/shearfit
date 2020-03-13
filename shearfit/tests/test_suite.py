import pdb
import esutil
import numpy as np
from unittest import TestCase
import astropy.units as units
import clusterlensing.nfw as cl
from nose.tools import set_trace
from ..analytic_profiles import NFW
from astropy.cosmology import WMAP7
import halotools.empirical_models as em
from ..lensing_system import obs_lens_system
from ..mass_concentration import child2018 as cm
from lenstronomy.GalKin.cosmo import Cosmo as lenstronomy


def _test_halo():
    '''
    Returns a set of arbitrary NFW halo parameters to use for the test provided here

    Returns
    -------
    halo : dictionary
        A set of NFW halo parameters, the radius `'r'`, concentration `'c'`, and redshift `'zl'`
    '''
    halo = {'r':2.0, 'c':3.7, 'zl':0.8}
    return halo


class TestNFW(TestCase):

    def test_radius_to_mass(self, cosmo=WMAP7, tolerance=1e-6):
        '''
        This functions tests the halo radius to mass conversion of the `NFW` class in 
        `analytic_profiles.py`, against the `halotools` package by Andrew Hearin

        Parameters
        ----------
        cosmo : `AstroPy` `cosmology` object
            The cosmology to use for computing density parameters
        tolerance : float
            The error tolerance to assert; if the fractional difference between this 
            package and `halotools` is above this value, then the test is failed.
        '''
        
        # decalre halo objects
        halo = _test_halo()
        this_NFW = NFW(halo['r'], halo['c'], halo['zl'])
        halotools_NFW = em.NFWProfile(cosmology=cosmo, redshift=halo['zl'], mdef='200c')

        # compute masses from radii
        this_m200c = this_NFW.radius_to_mass()
        halotools_m200c = halotools_NFW.halo_radius_to_halo_mass(halo['r'])

        # compute fractional difference and assert error tolerance
        fdiff = (this_m200c - halotools_m200c) / (halotools_m200c)
        self.assertTrue( fdiff <= tolerance)


    def test_delta_sigma(self, cosmo=WMAP7, tolerance=1e-6):
        '''
        This function test the differential surface density calculation of the `NFW` class in 
        `analytic_profiles.py`, against the `cluster-lensing` package by Jes Ford
        
        Parameters
        ----------
        cosmo : `AstroPy` `cosmology` object
            The cosmology to use for computing density parameters
        tolerance : float
            The error tolerance to assert; if the fractional difference between this 
            package and `halotools` is above this value, then the test is failed.
        '''
        
        # decalre halo objects and parameters
        halo = _test_halo()
        r_bins = np.linspace(0.1, halo['r']*3, 100)
        a = 1/(1+halo['zl'])
        
        # rho_crit in units that cluster-lensing needs
        rho_crit = cosmo.critical_density(halo['zl'])
        rho_crit = rho_crit.to(units.Msun/units.Mpc/units.pc**2).value
       
        this_NFW = NFW(halo['r'], halo['c'], halo['zl'])
        clusterlens_halo= cl.SurfaceMassDensity( rs = [this_NFW._rs],
                                                 delta_c = [this_NFW.del_c], 
                                                 rho_crit = [rho_crit], 
                                                 offsets= [0],
                                                 rbins = r_bins)
        
        # compute differential surface mass density
        # factor of a**2 in clusterlens_dsig to get a comoving surface area in pc^2
        this_dsig = this_NFW.delta_sigma(r_bins)
        clusterlens_dsig = clusterlens_halo.deltasigma_nfw().value[0] * a**2

        # compute fractional difference and assert error tolerance
        fdiff = (this_dsig - clusterlens_dsig) / (clusterlens_dsig)
        self.assertTrue( max(fdiff) <= tolerance)


    def test_sigma(self, cosmo=WMAP7, tolerance=1e-6):
        '''
        This function test the surface mass density calculation of the `NFW` class in 
        `analytic_profiles.py`, against the `cluster-lensing` package by Jes Ford
        
        Parameters
        ----------
        cosmo : `AstroPy` `cosmology` object
            The cosmology to use for computing density parameters
        tolerance : float
            The error tolerance to assert; if the fractional difference between this 
            package and `halotools` is above this value, then the test is failed.
        '''
        
        # decalre halo objects and parameters
        halo = _test_halo()
        r_bins = np.linspace(0.1, halo['r']*3, 100)
        a = 1/(1+halo['zl'])
        
        # rho_crit in units that cluster-lensing needs
        rho_crit = cosmo.critical_density(halo['zl'])
        rho_crit = rho_crit.to(units.Msun/units.Mpc/units.pc**2).value
       
        this_NFW = NFW(halo['r'], halo['c'], halo['zl'])
        clusterlens_halo= cl.SurfaceMassDensity( rs = [this_NFW._rs],
                                                 delta_c = [this_NFW.del_c], 
                                                 rho_crit = [rho_crit], 
                                                 offsets= [0],
                                                 rbins = r_bins)
        
        # compute surface mass density
        # factor of a**2 in clusterlens_dsig to get a comoving surface area in pc^2
        this_sigma = this_NFW.sigma(r_bins)
        clusterlens_sigma = clusterlens_halo.sigma_nfw().value[0] * a**2
        
        # compute fractional difference and assert error tolerance
        fdiff = (this_sigma - clusterlens_sigma) / (clusterlens_sigma)
        self.assertTrue( max(fdiff) <= tolerance)



    def test_sigma_crit(self, cosmo=WMAP7, tolerance=1e-3):
        '''
        This function test the critical surface density calculation of the `obs_lens_system` class 
        in `lensing_system.py`, against the `lenstronomy` package by Simon Birrer
        
        Parameters
        ----------
        cosmo : `AstroPy` `cosmology` object
            The cosmology to use for computing density parameters
        tolerance : float
            The error tolerance to assert; if the fractional difference between this 
            package and `halotools` is above this value, then the test is failed.
        '''
        
        # decalre halo objects and parameters
        halo = _test_halo()
        z_sources = np.linspace(halo['zl']+0.01, 2, 100)
        a = 1/(1+halo['zl'])
        this_lens = obs_lens_system(zl=halo['zl'])
        
        # compute critical surface density 
        # factor of a**2/(1e12) in birrer_sig_crit to get a comoving surface area in (pc)^2
        this_sig_crit = this_lens.calc_sigma_crit(zs = z_sources)
        birrer_sig_crit = np.zeros(len(z_sources))       
        for i in range(len(z_sources)):
            Dd = cosmo.angular_diameter_distance(halo['zl']).value
            Ds = cosmo.angular_diameter_distance(z_sources[i]).value
            Dds = Ds - Dd
            birrer_lens = lenstronomy(Dd, Ds, Dds)
            birrer_sig_crit[i] = birrer_lens.epsilon_crit * a**2 / (1e12)
        
        # compute fractional difference and assert error tolerance
        fdiff = (this_sig_crit - birrer_sig_crit) / (birrer_sig_crit)
        self.assertTrue( max(fdiff) <= tolerance)
