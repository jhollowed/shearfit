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
4) an NFW profile is fit to the scaled tangential shear ΔΣ = γ_t * Σ_c=
"""

# randomly place sources (x and y in arcsec)
x = (np.random.rand(75)-0.5) * 600
y = (np.random.rand(75)-0.5) * 600
zs = (np.random.rand(len(x))*0.5) + 0.8

# place lens (set shears to 0 for now)
zl = 0.8
r200 = 2.0
c = 3.0
mock_lens = obs_lens_system(zl)
mock_lens.set_background(x, y, zs, yt = np.zeros(len(x)))
bg = mock_lens.get_background()
r = bg['r']
sigmaCrit = mock_lens.calc_sigma_crit()

# assign all the sources with perfect NFW tangential shears, and add scatter (20% for 1std of pop)
rsamp = np.linspace(min(r), max(r), 1000)
true_profile = NFW(r200, c, zl)
dSigma_true = true_profile.delta_sigma(rsamp) 

dSigma_clean = true_profile.delta_sigma(r)
yt_clean = dSigma_clean/sigmaCrit
noise = np.sqrt(0.1) * np.random.randn(len(r))
yt_data = yt_clean + (yt_clean*noise)
mock_lens.yt = yt_data

# now let's go backward and fit these augmented shears to a new NFW profile, with guesses for
# the nfw parameters, assuming we have rough proiors on them
fitted_profile = NFW(1.0, 4.5, zl)
res_fit = fit(mock_lens, fitted_profile, rad_bounds = [0.5, 4], conc_bounds = [0.5, 6])
dSigma_fitted = fitted_profile.delta_sigma(rsamp)

# and now do it again, iteratively using a c-M relation instead of fitting for c
fitted_cm_profile = NFW(1.0, 4.5, zl)
res_cM = fit(mock_lens, fitted_cm_profile, rad_bounds = [0.5, 4], cM_relation='child2018')
dSigma_fitted_cm = fitted_cm_profile.delta_sigma(rsamp)

# no do a grid scan
gridscan_profile = NFW(1.0, 4.5, zl)
[grid_pos, grid_res] = fit_gs(mock_lens, gridscan_profile, rad_bounds = [0.01, 10], conc_bounds = [0.01, 10], n=200)

print('r200c_fit = {}; c_fit = {}'.format(fitted_profile.r200c, fitted_profile.c))
print('r200c_cm = {}; c_cm = {}'.format(fitted_cm_profile.r200c, fitted_cm_profile.c))

# vis shears 
rc('text', usetex=True)
color = plt.cm.plasma(np.linspace(0.2, 0.8, 3))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

f = plt.figure()
ax = f.add_subplot(111)
ax.plot(r, dSigma_clean, 'x', color='grey', label=r'$\gamma_{T,\mathrm{NFW}} \Sigma_c$')
ax.plot(r, yt_data*sigmaCrit, 'xk', label=r'$\gamma_{T,\mathrm{NFW}} \Sigma_c\>\> + \>\>\mathrm{Gaussian\>noise}$')
ax.plot(rsamp, dSigma_true, label=r'$\Delta\Sigma_\mathrm{{NFW}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'.format(
                                   true_profile.r200c, true_profile.c))
ax.plot(rsamp, dSigma_fitted, label=r'$\Delta\Sigma_\mathrm{{fit}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'.format(
                                    fitted_profile.r200c, fitted_profile.c))
ax.plot(rsamp, dSigma_fitted_cm, label=r'$\Delta\Sigma_{{\mathrm{{fit}},c-M}},\>\>r_{{200c}}={:.3f}; c={:.3f}$'.format(
                                       fitted_cm_profile.r200c, fitted_cm_profile.c))

ax.legend(fontsize=14, loc='upper right')
ax.set_xlabel(r'$r\>\>\lbrack\mathrm{Mpc}\rbrack$', fontsize=14)
ax.set_ylabel(r'$\Delta\Sigma\>\>\lbrack\mathrm{M}_\odot\mathrm{pc}^{-2}\rbrack$', fontsize=14)

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.scatter(grid_pos[0], grid_pos[1], c=1/grid_res, cmap='plasma')

plt.tight_layout()
plt.show()
