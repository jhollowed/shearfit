import pdb
import numpy as np
import matplotlib.pyplot as plt
from analytic_profiles import NFW
from lensing_system import obs_lens_system
from fit_profile import fit_nfw_profile_lstq as fit

"""
This script performs an example run of the package, fitting an NFW profile to synthetically 
generated background source data. The process is as follows:
1) sources are placed randomly in (theta1, theta2, z) space
2) sources are assigned tangential shears analytically, given an NFW assumption
3) gaussian scatter is added to the shear values
4) an NFW profile is fit to the scaled tangential shear ΔΣ = γ_t * Σ_c=
"""

# randomly place sources
x = (np.random.rand(50)-0.5) * 6
y = (np.random.rand(50)-0.5) * 6
r = np.linalg.norm([x, y], axis=0)
zs = (np.random.rand(len(r))*0.5) + 0.8

# place lens (set shears to 0 for now)
zl = 0.8
r200 = 2.0
c = 3.0
mock_lens = obs_lens_system(zl)
mock_lens.set_background(x, y, zs, yt = 0)
sigmaCrit = mock_lens.calc_sigma_crit()

# assign all the sources with perfect NFW tangential shears, and add scatter (20% for 1std of pop)
profile = NFW(r200, c, zl)
dSigma_true = profile.delta_sigma(np.linspace(min(r), max(r), 1000)) 
dSigma_data = profile.delta_sigma(r)

yt_clean = dSigma_data/sigmaCrit
noise_data = np.sqrt(0.1) * np.random.randn(len(r))
yt_data = yt_clean + (yt_clean*noise_data)
mock_lens.yt = yt_data

# now let's go backward and fit these augmented shears to the NFW profile we defined above,
# after scrambling it's parameters, and assume we have rough proiors on the radius and concentration
profile.r200c = 1.0
profile.c = 4.5
res_fit = fit(mock_lens, profile, rad_bounds = [0.5, 4], conc_bounds = [0.5, 6])

# and now do it again, iteratively using a c-M relation instead of fitting for c
res_cM = fit(mock_lens, profile, rad_bounds = [0.5, 4], cM_relation='child2018')

# vis shears 
rorder = np.argsort(r)
f = plt.figure()
ax = f.add_subplot(111)
ax2 = ax.twinx()
l1 = ax.plot(r[rorder], (y*sc)[rorder], '-xr', markeredgecolor='k', markerfacecolor='k', label='γ_t*Σ_c')
#ax.plot(r[rorder], (y*sc)[rorder], 'xk', label='')
l2 = ax2.plot(r[rorder], y[rorder], '+', color='grey', label='γ_t')
ax.set_yscale('log')
ax2.set_yscale('log')
ax.set_xscale('log')

lns = l1+l2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='lower left')

ax.set_xlabel('r [Mpc]')
ax.set_ylabel('ΔΣ [M_sun/Mpc^2]')
ax2.set_ylabel('γ_t')
plt.tight_layout()

plt.show()



