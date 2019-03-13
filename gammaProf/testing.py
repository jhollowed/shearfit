import numpy as np
from analytic_profiles import NFW_shear_profile as nfw
import matplotlib.pyplot as plt

# randomly place sources
x = (np.random.rand(50)-0.5) * 6
y = (np.random.rand(50)-0.5) * 6
r = np.linalg.norm([x, y], axis=0)

zs = (np.random.rand(len(r))*0.5) + 0.8
#zs = np.ones(len(r))*1.0

# vis sources
#plt.scatter(x, y, color='r',  s=(zs**2)*2)
#plt.show()

# place lens
zl = 0.8
r200 = 2.0
c = 3.0

# calc shears

prof = nfw(r200, c)
y = prof.scaled_prediction(r, zs, zl)


# vis shears 

rorder = np.argsort(r)
plt.plot(r[rorder], y[rorder], '-r')
plt.plot(r[rorder], y[rorder], 'xk')
plt.yscale('log')
plt.xscale('log')

plt.show()



