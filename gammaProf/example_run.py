import pdb
import cycler
import numpy as np
import matplotlib.pyplot as plt
from analytic_profiles import NFW
from lensing_system import obs_lens_system
from fit_profile import fit_nfw_profile_lstq as fit
from fit_profile import fit_nfw_profile_gridscan as fit_gs
from matplotlib import rc
import matplotlib as mpl

"""
This script performs an example run of the package, fitting an NFW profile to synthetically 
generated background source data. The process is as follows:
1) sources are placed randomly in (theta1, theta2, z) space
2) sources are assigned tangential shears analytically, given an NFW assumption
3) gaussian scatter is added to the shear values
4) an NFW profile is fit to the scaled tangential shear ΔΣ = γ_t * Σ_c, using three methods:
    - allow both the concentration and halo radius to float
    - only fit the halo radius, inferring the concentration from a c-M relation
    - do a grid scan over the parameter pair including radius and concentration
"""

# randomly place sources (x and y in arcsec)
print('generating data')
x = (np.random.rand(75)-0.5) * (750*2)
y = (np.random.rand(75)-0.5) * (750*2)
zs = (np.random.rand(len(x))*0.5) + 0.8

# place lens (set shears to 0 for now)
zl = 0.35
r200 = 4.25
c = 4.0
mock_lens = obs_lens_system(zl)
mock_lens.set_background(x, y, zs, yt = np.zeros(len(x)))
bg = mock_lens.get_background()
r = bg['r']
sigmaCrit = mock_lens.calc_sigma_crit()

# assign all the sources with perfect NFW tangential shears, and add scatter (10% for 1std of pop)
rsamp = np.linspace(min(r), max(r), 1000)
true_profile = NFW(r200, c, zl)
dSigma_true = true_profile.delta_sigma(rsamp)

dSigma_clean = true_profile.delta_sigma(r)
yt_clean = dSigma_clean/sigmaCrit
noise = (np.sqrt(0.1) * np.random.randn(len(r)))
yt_data = yt_clean + (yt_clean*noise)
mock_lens.yt = yt_data

# now let's go backward and fit these augmented shears to a new NFW profile, with guesses for
# the nfw parameters, assuming we have rough proiors on them
print('fitting with floating concentration')
fitted_profile = NFW(2.0, 2.0, zl)
[res_fit, fit_err] = fit(mock_lens, fitted_profile, rad_bounds = [1, 6], conc_bounds = [2, 8], 
                         bootstrap=True)
dSigma_fitted = fitted_profile.delta_sigma(rsamp)
# get error band
fitted_profile.r200c = fitted_profile.r200c + fit_err[0]
fitted_profile.c = fitted_profile.c + fit_err[1]
dSigma_fitted_err0 = fitted_profile.delta_sigma(rsamp)
fitted_profile.r200c = fitted_profile.r200c - 2*fit_err[0]
fitted_profile.c = fitted_profile.c - 2*fit_err[1]
dSigma_fitted_err1 = fitted_profile.delta_sigma(rsamp)

# and now do it again, iteratively using a c-M relation instead of fitting for c
print('fitting with inferred c-M concentration')
fitted_cm_profile = NFW(2.0, 2.0, zl)
[res_cm_fit, cm_fit_err] = fit(mock_lens, fitted_cm_profile, rad_bounds = [1, 6], 
                           cM_relation='child2018', bootstrap=True)
dSigma_fitted_cm = fitted_cm_profile.delta_sigma(rsamp)
# get error band
fitted_cm_profile.r200c = fitted_cm_profile.r200c + cm_fit_err[0]
fitted_cm_profile.c = fitted_cm_profile.c + cm_fit_err[1]
dSigma_fitted_cm_err0 = fitted_cm_profile.delta_sigma(rsamp)
fitted_cm_profile.r200c = fitted_cm_profile.r200c - 2*cm_fit_err[0]
fitted_cm_profile.c = fitted_cm_profile.c - 2*cm_fit_err[1]
dSigma_fitted_cm_err1 = fitted_cm_profile.delta_sigma(rsamp)

# now do a grid scan
print('doing grid scan')
gridscan_profile = NFW(2.0, 2.0, zl)
[grid_pos, grid_res] = fit_gs(mock_lens, gridscan_profile, rad_bounds = [1, 6], conc_bounds = [2, 8], n=200)

print('r200c_fit = {}; c_fit = {}'.format(fitted_profile.r200c, fitted_profile.c))
print('r200c_cm = {}; c_cm = {}'.format(fitted_cm_profile.r200c, fitted_cm_profile.c))

# visualize results 
rc('text', usetex=True)
color = plt.cm.plasma(np.linspace(0.2, 0.8, 3))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

f = plt.figure(figsize=(12,6))
ax = f.add_subplot(121)
ax.plot(r, dSigma_clean, 'x', color='grey', label=r'$\gamma_{T,\mathrm{NFW}} \Sigma_c$')
ax.plot(r, yt_data*sigmaCrit, 'xk', label=r'$\gamma_{T,\mathrm{NFW}} \Sigma_c\>\> + \>\>\mathrm{Gaussian\>noise}$')
ax.plot(rsamp, dSigma_true, '--', label=r'$\Delta\Sigma_\mathrm{{NFW}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'\
                                        .format(r200, c), color=color[0], lw=2)
ax.plot(rsamp, dSigma_fitted, label=r'$\Delta\Sigma_\mathrm{{fit}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'\
                                    .format(res_fit.x[0], res_fit.x[1]), color=color[1], lw=2)
ax.plot(rsamp, dSigma_fitted_cm, label=r'$\Delta\Sigma_{{\mathrm{{fit}},c-M}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'\
                                       .format(res_cm_fit.x[0], fitted_cm_profile.c + cm_fit_err[1]), 
                                       color=color[2], lw=2)
ax.fill_between(rsamp, dSigma_fitted_err0, dSigma_fitted_err1, color=color[1], alpha=0.2, lw=0)
ax.fill_between(rsamp, dSigma_fitted_cm_err0, dSigma_fitted_cm_err1, color=color[2], alpha=0.33, lw=0)

ax.legend(fontsize=14, loc='upper right')
ax.set_xlabel(r'$r\>\>\lbrack\mathrm{Mpc}\rbrack$', fontsize=14)
ax.set_ylabel(r'$\Delta\Sigma\>\>\lbrack\mathrm{M}_\odot\mathrm{pc}^{-2}\rbrack$', fontsize=14)

ax2 = f.add_subplot(122)
pdb.set_trace()
chi2 = ax2.pcolormesh(grid_pos[0], grid_pos[1], (1/grid_res)/(np.max(1/grid_res)), cmap='plasma')
ax2.plot([r200], [c], 'xk', ms=10, label=r'$\Delta\Sigma_\mathrm{{true}}$')
ax2.errorbar(res_fit.x[0], res_fit.x[1], xerr=fit_err[0], yerr=fit_err[1], ms=10, marker='.', c='c',
             label=r'$\Delta\Sigma_\mathrm{{fit}}$')
ax2.errorbar(res_cm_fit.x[0], fitted_cm_profile.c + cm_fit_err[1], 
             xerr=cm_fit_err[0], yerr=cm_fit_err[1], ms=10, marker='.', c='b',
             label=r'$\Delta\Sigma_{\mathrm{{fit},c\mathrm{-}M}}$')
ax2.legend(fontsize=14, loc='upper right')
cbar = f.colorbar(chi2, ax=ax2)
cbar.set_label(r'$\left[(\chi^2/\mathrm{min}(\chi^2))\right]^{-1}$', fontsize=14)
ax2.set_xlabel(r'$r_{200c}\>\>\left[\mathrm{Mpc}\right]$', fontsize=14)
ax2.set_ylabel(r'$c_{200c}$', fontsize=14)

plt.tight_layout()
plt.show()
