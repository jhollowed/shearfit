import numpy as np
from analytic_profiles import NFW
import matplotlib.pyplot as plt
from lens import lens
import pdb

# randomly place sources
x = (np.random.rand(50)-0.5) * 6
y = (np.random.rand(50)-0.5) * 6
r = np.linalg.norm([x, y], axis=0)

zs = (np.random.rand(len(r))*0.5) + 0.8
#zs = np.ones(len(r))*1.0

# place lens
zl = 0.8
test_lens = lens(zl)
test_lens.set_background(x, y, zs)
sc = test_lens.calc_sigma_crit()

# calc shears
r200 = 2.0
c = 3.0
prof = NFW(r200, c, zl)
ds = prof.delta_sigma(r)
y = ds/sc

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



