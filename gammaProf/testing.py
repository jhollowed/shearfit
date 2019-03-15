import numpy as np
from analytic_profiles import NFW
import matplotlib.pyplot as plt
from lensing_system import obs_lens_system
import pdb

# randomly place sources
x = (np.random.rand(50)-0.5) * 6
y = (np.random.rand(50)-0.5) * 6
r = np.linalg.norm([x, y], axis=0)
zs = (np.random.rand(len(r))*0.5) + 0.8

# place lens
zl = 0.8
test_lens = obs_lens_system(zl)
test_lens.set_background(x, y, zs)
sc = test_lens.calc_sigma_crit()

# let's assign all the sources with perfect NFW tang. shears, and add some noise (0-20%)
r200 = 2.0
c = 3.0
prof = NFW(r200, c, zl)
ds_true = prof.delta_sigma(np.linspace(min(r), max(r), 1000)) 

ds_data = prof.delta_sigma(r)
y_clean = ds/sc
noise_data = np.random.rand(len(r)) * (np.mean(y_data) * 0.33)
y_data = y_clean * noise_data

# now let's go backward and fit these augmented shears to an NFW profile




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



