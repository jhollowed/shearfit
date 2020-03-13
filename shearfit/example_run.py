import os
import pdb
import sys
import h5py
import glob
import cycler
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from analytic_profiles import NFW
from astropy.cosmology import WMAP7
from lensing_system import obs_lens_system
from mass_concentration import child2018 as cm
from fit_profile import fit_nfw_profile_lstq as fit
from fit_profile import fit_nfw_profile_gridscan as fit_gs

pprint = lambda s: print(s, flush=True)

def mock_example_run(zl=0.35, r200c=2, c=3.7, nsources=1000, fov=1500, z_dls=1.0, noisef=0.1, bin_data=False):
    """
    This function performs an example run of the package, fitting an NFW profile to synthetically 
    generated background source data. The process is as follows:
    1) sources are placed randomly in (theta1, theta2, z) space
    2) sources are assigned tangential shears analytically, given an NFW assumption
    3) gaussian scatter is added to the shear values
    4) an NFW profile is fit to the scaled tangential shear ΔΣ = γ_t * Σ_c, using three methods:
        - allow both the concentration and halo radius to float
        - only fit the halo radius, inferring the concentration from a c-M relation
        - do a grid scan over the parameter pair including radius and concentration
    
    Parameters
    ----------
    zl : float
        The lens redshift. Defaults to `0.35`.
    r200c : float
        The :math:`r_{200c}` radius of the lens, in Mpc. Defaults to `4.25`.
    c : float
        The dimensionless NFW concentration of the lens. Defaults to `4.0`.
    nsouces : int
        The number of sources to pace in the background. Defaults to `75`.
    fov : float 
        The side length of the field of view, in arcseconds. Defaults to `1500`.
    z_dls : float
        The maximum redshift difference between the lens and sources. Source 
        redshifts will be randomly drawn from a uniform distribution on [`zl`, `zl+z_dls`].
        Defaults to `1.0`.
    noisef : float
        The amount of scatter to add to the mock data. Specifically, the clean analytic 
        signal for one data point will be modified by a gaussian smoothing as
        `data = clean_signal * (np.sqrt(noisef) * np.random.randn(1) + 1)`.
        Defaults to `0.1`.
    bin_data : bool
        whether or not to fit to shears averaged in radial bins, rather than to each individual source.
        Defaults to `False`.
    """

    [mock_lens, true_profile] = _gen_mock_data(zl, r200c, c, nsources, fov, z_dls, noisef=noisef)
    _fit_test_data(mock_lens, true_profile, showfig=True, bin_data=bin_data)


def sim_example_run(halo_cutout_dir='/projects/DarkUniverse_esp/jphollowed/outerRim/cutouts_raytracing/'\
                                     'halo_4781763152100605952_0', 
                    makeplot = True, showfig=True, stdout=True, bin_data=True, rbins=25, rmin=0):
    """
    This function performs an example run of the package, fitting an NFW profile to background 
    source data as obtained from ray-tracing through Outer Rim lightcone halo cutouts. The process 
    is as follows:
    1) sources positions, redshifts, and shears are read in from ray-tacing outputs
    4) an NFW profile is fit to the scaled tangential shear ΔΣ = γ_t * Σ_c, using three methods:
        - allow both the concentration and halo radius to float
        - only fit the halo radius, inferring the concentration from a c-M relation
        - do a grid scan over the parameter pair including radius and concentration
    
    Parameters
    ----------
    halo_dir : string
        Path to a halo cutout ray-tacing output directory. If `None`, then randomly select a simulated
        halo from those available on the filesystem (assumes the code to be running on Cori at NERSC). 
        This directory is assumed to contain an HDF5 file giving ray-traced maps, as well as a properties.csv
        file, containing the intrinsic halo properties from the simulation.
    max_z : float
        The maximum number of lens planes used about the halo redshift in the ray tracing
    makeplot : boolean
        Whether or not to make render a plot. If `True`, perform a  grid scan in addition to the 
        two profile fits to plot residuals over parameter space. If `False`, return after fit and
        skip parameter sweep, plot nothing. Defaults to `True`.
    showfig : boolean
        Whether or not to `show` the profile fit plot. If `False`, save to `png` file instead. 
        Defaults to `True`.
    stdout : bool
        Whether or not to supress print statements (useful for parallel runs). `False` means all
        print statements will be suppressed. Defaults to `True`.
    bin_data : bool
        whether or not to fit to shears averaged in radial bins, rather than to each individual source.
        Defaults to `True`.
    rbins : int
        Number of bins distribute over the radial range of the data, if `bin_data == True`.
    rmin : float
        The minimum radial distance of sources to include in the fit (e.g. `rmin = 0.3`) will
        remove the inner 300kpc of source information. Defaults to `0`.
    """
    
    global pprint
    if(stdout == True): pprint = lambda s: print(s, flush=True)
    else: pprint = lambda s: None

    pprint('reading lensing mock data')
    [sim_lens, true_profile] = _read_sim_data(halo_cutout_dir)
    out_dir = '{}/profile_fits'.format(halo_cutout_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    _fit_test_data(sim_lens, true_profile, showfig=showfig, 
                   out_dir=out_dir, makeplot = makeplot, bin_data=bin_data, 
                   rbins=rbins, rmin=rmin)


def _gen_mock_data(zl, r200c, c, nsources, fov, z_dls, noisef=0.1):

    # randomly place sources (x and y in arcsec)
    print('generating data')
    x = (np.random.rand(nsources)-0.5) * (fov)
    y = (np.random.rand(nsources)-0.5) * (fov)
    zs = (np.random.rand(len(x)) * z_dls) + zl

    # place lens (set shears to 0 for now)
    mock_lens = obs_lens_system(zl)
    mock_lens.set_background(x, y, zs, yt = np.zeros(len(x)), k = np.zeros(len(x)))

    # assign all the sources with NFW-implied tangential shears, and add scatter (10% for 1std of pop)
    sigmaCrit = mock_lens.calc_sigma_crit()
    bg = mock_lens.get_background()
    r = bg['r']
    
    true_profile = NFW(r200c, c, zl)
    dSigma_clean = true_profile.delta_sigma(r)
    sigma_clean = true_profile.sigma(r)

    yt_clean = dSigma_clean / sigmaCrit
    noise = (np.sqrt(noisef) * np.random.randn(len(r)))
    yt_data = yt_clean + (yt_clean*noise)
    mock_lens.yt = yt_data

    k_clean = sigma_clean / sigmaCrit
    noise = (np.sqrt(noisef) * np.random.randn(len(r)) + 1)
    k_data = k_clean * noise
    mock_lens.k = k_data

    return [mock_lens, true_profile]
    

def _read_sim_data(halo_cutout_dir):

    # get ray-trace hdf5 and properties csv
    rtfs = glob.glob('{}/*lensing_mocks.hdf5'.format(halo_cutout_dir))
    
    #YYY rtfs = glob.glob('{}/*gmaps.hdf5'.format(halo_cutout_dir))
    
    pfs = glob.glob('{}/properties.csv'.format(halo_cutout_dir))
    assert len(pfs) == 1, "Exactly one properties file is expected in {}".format(halo_cutout_dir)

    # read lens properties from csv, source data from hdf5
    props_file = pfs[0]
    props = np.genfromtxt(props_file, delimiter=',', names=True)
    zl = props['halo_redshift']
    r200c = props['sod_halo_radius']
    c = props['sod_halo_cdelta']
    c_err = props['sod_halo_cdelta_error']
    m200c = props['sod_halo_mass']
    true_profile = NFW(r200c, c, zl, c_err = c_err)
    
    raytrace_file = h5py.File(rtfs[0], 'r')
    nplanes = len(list(raytrace_file.keys()))
    t1, t2, y1, y2, k, zs = [], [], [], [], [], []

    # stack data from each source plane
    for i in range(nplanes):
        plane_key = list(raytrace_file.keys())[i]
        plane = raytrace_file[plane_key]
        plane_z = plane['zs'][:]

        #ignore this plane if infront of the halo
        #YYY if(i > 0): continue
        if(plane_z < zl): continue
       
        t1 = np.hstack([t1, plane['xr1'][:]])
        t2 = np.hstack([t2, plane['xr2'][:]])
        y1 = np.hstack([y1, plane['sr1'][:]])
        y2 = np.hstack([y2, plane['sr2'][:]])
        k = np.hstack([k, plane['kr0'][:]])
        zs = np.hstack([zs, np.ones(len(t1)-len(zs)) * plane_z])
        
        #YYY y1 = np.hstack([y1, np.ravel(plane['shear1'][:])])
        #YYY y2 = np.hstack([y2, np.ravel(plane['shear2'][:])])
        #YYY k = np.hstack([k, np.ravel(plane['kappa0'][:])])
        #YYY T1 = np.linspace(-props['boxRadius_arcsec'], props['boxRadius_arcsec'], 1024)
        #YYY T2 = np.linspace(-props['boxRadius_arcsec'], props['boxRadius_arcsec'], 1024)
        #YYY grid = np.meshgrid(T1, T2)
        #YYY t1 = np.ravel(grid[0])
        #YYY t2 = np.ravel(grid[1])
        #YYY zs = np.hstack([zs, np.ones(len(t1)-len(zs)) * 10])

     
    # trim the fov borders by 10% to be safe
    mask = np.logical_and(np.abs(t1)<props['boxRadius_arcsec']*0.9, 
                          np.abs(t2)<props['boxRadius_arcsec']*0.9)
    t1 = t1[mask]
    t2 = t2[mask]
    zs = zs[mask]
    y1 = y1[mask]
    y2 = y2[mask]
    k = k[mask]
    
    sim_lens = obs_lens_system(zl)
    sim_lens.set_background(t1, t2, zs, y1=y1, y2=y2, k=k)

    raytrace_file.close()
    
    return [sim_lens, true_profile]

    
def _fit_test_data(lens, true_profile, makeplot=True, showfig=False, out_dir='.', 
                   bin_data=True, rbins=25, rmin = 0):

    zl = lens.zl
    r200c = true_profile.r200c
    c = true_profile.c
    bg = lens.get_background()
    sigmaCrit = lens.calc_sigma_crit()
    yt = bg['yt']
    r = bg['r']
    k = bg['k']
    zs = bg['zs']
    
    #if(zl == 0.831494): pass
    #else: return

    # do inner radius cut and bin data
    pprint('doing radial masking and binning')
    radial_mask = (r >= rmin)
    sigmaCrit = sigmaCrit[radial_mask]
    yt = yt[radial_mask]
    r = r[radial_mask]
    k = k[radial_mask]
    zs = zs[radial_mask]
    binned_dsig = stats.binned_statistic(r, yt*sigmaCrit, statistic='mean', bins=rbins)
    binned_r = stats.binned_statistic(r, r, statistic='mean', bins=rbins)

    rsamp = np.linspace(min(r), max(r), 1000)
    dSigma_true = true_profile.delta_sigma(rsamp)

    e = true_profile.sigma(r)
    se = true_profile.delta_sigma(r)
    kk = k * sigmaCrit
    binned_kk = stats.binned_statistic(r, kk, statistic='mean', bins=rbins)
    binned_yt = stats.binned_statistic(r, yt, statistic='mean', bins=rbins) 

    # fit the concentration and radius
    pprint('fitting with floating concentration')
    fitted_profile = NFW(0.75, 3.0, zl)
    fit(lens, fitted_profile, rad_bounds = [0.1, 15], conc_bounds = [1, 10], 
        bootstrap=True, bin_data=bin_data, bins=rbins)
    #[dSigma_fitted, dSigma_fitted_err] = fitted_profile.delta_sigma(rsamp, bootstrap=True)
    dSigma_fitted = fitted_profile.delta_sigma(rsamp, bootstrap=False)
    dSigma_fitted_err = np.zeros(len(dSigma_fitted))

    # and now do it again, iteratively using a c-M relation instead of fitting for c
    pprint('fitting with inferred c-M concentration')
    fitted_cm_profile = NFW(0.75, 3.0, zl)
    fit(lens, fitted_cm_profile, rad_bounds = [0.1, 15], cM_relation='child2018', 
        bootstrap=True, bin_data=bin_data, bins=rbins)
    #[dSigma_fitted_cm, dSigma_fitted_cm_err] = fitted_cm_profile.delta_sigma(rsamp, bootstrap=True)
    dSigma_fitted_cm = fitted_profile.delta_sigma(rsamp, bootstrap=False)
    dSigma_fitted_cm_err = np.zeros(len(dSigma_fitted_cm))
    
    # write out fitting result
    pprint('r200c_fit = {}; c_fit = {}'.format(fitted_profile.r200c, fitted_profile.c))
    pprint('r200c_cm = {}; c_cm = {}'.format(fitted_cm_profile.r200c, fitted_cm_profile.c))
    np.save('{}/r200c_fit_{}bins_{}rmin.npy'.format(out_dir, rbins, rmin), fitted_profile.r200c)
    np.save('{}/r200c_cM_fit_{}bins_{}rmin.npy'.format(out_dir, rbins, rmin), fitted_cm_profile.r200c)
    np.save('{}/c_fit_{}bins_{}rmin.npy'.format(out_dir, rbins, rmin), fitted_profile.c)
    np.save('{}/c_cM_fit_{}bins_{}rmin.npy'.format(out_dir, rbins, rmin), fitted_cm_profile.c)

    # all done if not plotting
    #RRR if(not makeplot): return

    # now do a grid scan
    pprint('doing grid scan')
    gridscan_profile = NFW(0.75, 3.0, zl)
    grid_r_bounds = [0.1, 4]
    grid_c_bounds = [1, 10]
    [grid_pos, grid_res] = fit_gs(lens, gridscan_profile, rad_bounds = grid_r_bounds, conc_bounds = grid_c_bounds, 
                                  n=200, bin_data=bin_data, bins=rbins)
    
    # visualize results... 
    rc('text', usetex=True)
    color = plt.cm.plasma(np.linspace(0.2, 0.8, 3))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    # plot sources vs truth and both fits
    f = plt.figure(figsize=(12,6))
    ax = f.add_subplot(121)
    
    if(not bin_data):
        ax.plot(r, yt*sigmaCrit, 'xk', 
                label=r'$\gamma_{T,\mathrm{NFW}} \Sigma_c\>\>+\>\>\mathrm{Gaussian\>noise}$', alpha=0.33)
    else:
        ax.plot(binned_r[0], binned_dsig[0], '-xk', 
                label=r'$\gamma_{T,\mathrm{NFW}} \Sigma_c\>\>+\>\>\mathrm{Gaussian\>noise}$')
    ax.plot(rsamp, dSigma_true, '--', label=r'$\Delta\Sigma_\mathrm{{NFW}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'\
                                            .format(r200c, c), color=color[0], lw=2)
    ax.plot(rsamp, dSigma_fitted, label=r'$\Delta\Sigma_\mathrm{{fit}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'\
                                        .format(fitted_profile.r200c, fitted_profile.c), color=color[1], lw=2)
    ax.plot(rsamp, dSigma_fitted_cm, label=r'$\Delta\Sigma_{{\mathrm{{fit}},c-M}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'\
                                           .format(fitted_cm_profile.r200c, fitted_cm_profile.c), 
                                           color=color[2], lw=2)
    ax.fill_between(rsamp, dSigma_fitted - dSigma_fitted_err.T[0], 
                           dSigma_fitted + dSigma_fitted_err.T[1], 
                           color=color[1], alpha=0.2, lw=0)
    ax.fill_between(rsamp, dSigma_fitted_cm - dSigma_fitted_cm_err.T[0], 
                           dSigma_fitted_cm + dSigma_fitted_cm_err.T[1], 
                           color=color[2], alpha=0.33, lw=0)

    # format
    ax.legend(fontsize=14, loc='upper right')
    ax.set_xlabel(r'$r\>\>\lbrack\mathrm{Mpc}\rbrack$', fontsize=14)
    ax.set_ylabel(r'$\Delta\Sigma\>\>\lbrack\mathrm{M}_\odot\mathrm{pc}^{-2}\rbrack$', fontsize=14)


    # plot fit cost in the radius-concentration plane
    ax2 = f.add_subplot(122)
    chi2 = ax2.pcolormesh(grid_pos[0], grid_pos[1], (1/grid_res)/(np.max(1/grid_res)), cmap='plasma')
    ax2.plot([r200c], [c], 'xk', ms=10, label=r'$\mathrm{{truth}}$')
    ax2.errorbar(fitted_profile.r200c, fitted_profile.c, 
                 xerr=fitted_profile.r200c_err, yerr=fitted_profile.c_err, 
                 ms=10, marker='.', c=color[1], label=r'$\mathrm{{fit}}$')
    ax2.errorbar(fitted_cm_profile.r200c, fitted_cm_profile.c, 
                 xerr=fitted_cm_profile.r200c_err, yerr=fitted_cm_profile.c_err, 
                 ms=10, marker='.', c=color[2], label=r'${\mathrm{{fit\>w/}c\mathrm{-}M}}$')

    # include c-M relation curve
    tmp_profile = NFW(1,1,zl)
    tmp_m200c = np.zeros(len(grid_pos[0][0]))
    for i in range(len(tmp_m200c)):
        tmp_profile.r200c = grid_pos[0][0][i]
        tmp_m200c[i] = tmp_profile.radius_to_mass()
    tmp_c, tmp_dc = cm(tmp_m200c)
    ax2.plot(grid_pos[0][0], tmp_c, '--k', lw=2, label=r'$c\mathrm{-}M\mathrm{\>relation\>(Child+2018)}$')
    ax2.fill_between(grid_pos[0][0], tmp_c - tmp_dc, tmp_c + tmp_dc, color='k', alpha=0.1, lw=0)

    # format
    ax2.set_xlim(grid_r_bounds)
    ax2.set_ylim(grid_c_bounds)
    ax2.legend(fontsize=14, loc='upper right')
    cbar = f.colorbar(chi2, ax=ax2)
    cbar.set_label(r'$\left[(\chi^2/\mathrm{min}(\chi^2))\right]^{-1}$', fontsize=14)
    ax2.set_xlabel(r'$r_{200c}\>\>\left[\mathrm{Mpc}\right]$', fontsize=14)
    ax2.set_ylabel(r'$c_{200c}$', fontsize=14)

    plt.tight_layout()
    plt.show()
    #RRR if(showfig): plt.show()
    #RRR else: f.savefig('{}/{}_shearprof_fit_{}bins_{}rmin.png'.format(out_dir, zl, rbins, rmin), dpi=200)

if(__name__ == "__main__"): 
    mock_example_run()
