import numpy as np
from lensing_system import obs_lens_system
from analytic_profile import NFW
from scipy import optimize
from mass_concentration import child2018

def fit_nfw_profile_lstq(data, profile, rad_bounds, conc_bounds = [0,10], cM_relation=None):
    """
    Fits an NFW-predicted :math:`\\Delta\\Sigma(r)` profile to a background shear dataset. To use
    this function, the user should first instantiate a `obs_lens_system` object, which will hold the
    observed data to fit to, and an `NFW` object, which will describe the analytic form which should
    describe the data. The present function is then the mediator between these objects that will
    facilitate the minimization routine.

    Parameters
    ----------
    data : `obs_len_system` class instance
        An instance of a `obs_lens_system` object as provided by `lensing_system.py`. 
        This is an object representing a lensing system, and contains data vectors 
        describing properties of a cluster's background sources
    profile : `NFW` class instance
        An instance of a `NFW` object as provided by `analytic_profiles.py`. This is
        an object representing an analytic NFW profile, and computes the predicted 
        projected surface density
    rad_bounds : 2-element list
        The bounds (tophat prior) for the first fitting parameter, :math:`r_{200c}`
    conc_bounds : 2-element list, optional
        The bounds (tophat prior) for the second fitting parameter, :math:`c`. Defaults to [0,10]
    cM_relation : string, optional
        The name of a :math:`c-M` relation to use in the fitting procedure. If `None`, then the 
        minimization will proceed with respect to both :math:`r_200c` and :math:`c`. If provided
        as a `string`, then infer the concentration from the :math:`c-M` relation on each iteration of 
        the least squares routine (in this case, the `conc_bounds` arg need not be passed). Options are 
        `{'child2018'}`. Defaults to `None`.

    Returns
    -------
    SciPy `OptimizeResult` object
        Fields are defined as detailed in the return signature of `scipy.optimize.least_squares`. 
        See documentation here: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """
    
    # get the background data, and scale the tangential shear to ΔΣ 
    [theta1, theta2, r, zs, _, _, yt] = data.get_background()
    Ec = data.calc_sigma_crit()
    dSigma_data = yt * Ec

    # get parameter guesses from initial NFW form
    rad_init = profile.r200c()
    conc_init = profile.c()

    # initiate the fitting algorithm
    if(cM_relation is None):
        fit_params = [rad_init, conc_init]
        bounds = ([rad_bounds[0], conc_bounds[0],                                                                                        rad_bounds[1], conc_bounds[1]])
    else:
        fit_params = [rad_init]
        bounds = ([rad_bounds[0], rad_bounds[1])
    
    res = optimize.least_squares(nfw_fit_residual, fit_params, 
                                 args=(profile, r, dSigma_data, cM_relation), 
                                 bounds = bounds)


    
def _nfw_fit_residual(fit_params, profile, r, dSigma_data, cM_relation)
    """
    Evaluate the residual of an NFW profile fit to data, given updated parameter values. 
    This function meant to be called iteratively from `fit_nfw_profile_lstq` only.

    Parameters
    ----------
    fit_params : float list
        The NFW parameter(s) to update for this least squares iteration 
        (either a single-element list including the radius, [r200c], or also including
        the concentration parameter=, [r200c, c])
    profile : `NFW` class instance
        An instance of a `NFW` object as provided by `analytic_profiles.py`. This is the 
        object that is being augmented per-iteration in the calling least squares routine 
    r : float array
        The halo-centric radial distances for the sources whose shears are given by yt_data
        (the NFW profile fitting form will be evaluated at these locations)
    dSigma_data : float array
        tangential shear values for a collection of background sources, against which to fit
        the profile
    cM_relation:
        The name of a :math:`c-M` relation to use in the fitting procedure. If `None`, then the 
        minimization will proceed with respect to both :math:`r_200c` and :math:`c`. If provided
        as a `string`, then infer the concentration from the :math:`c-M` relation on each iteration of 
        the least squares routine. Options are `{'child2018'}`.

    Returns
    -------
    residuals : float array
        The residuals between the data `dSigma_data`, and the NFW result given the `fit_params`, `dSigma_nfw`. 
        The residuals, in this case, are the difference `dSigma_data - dSigma_nfw`.
    """
   
    # update the NFW analytical profile object
    if(len(fit_params) > 1):
        
        # concentration being fitted
        r200c, c = fit_params[0], fit_params[1]
        profile.r200c = r200c
        profile.c = c
    
    else:
        
        # concentration modeled from c-M relation
        cM_func = {'child+2018':child+2018}[cM_relation]
        r200c = fit_params[0]
        m200c = profile.radius_to_mass()
        c_new = child2018(m200c)
        
        profile.r200c = r200c
        profile.c = c_new
    
    # evaluate NFW form
    dSigma_nfw = profle.delta_sigma(r)
    
    # residuals
    residuals = dSigma_data - dSigma_nfw
    return residuals 
