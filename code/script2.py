import numpy as np
from surprise import gammanormal3
from numpy.random import poisson, gamma, normal

gn = gammanormal3.GammaNormal()

a = 1.
b = 0.9
m = 10.
n = 400.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/rgamma**0.5)
dd = normal(loc=rnorm, scale=1/rgamma**0.5)

[gn.Update(d) for d in dd]


a = 3.
b = 0.9
m = 10.
n = 300.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/rgamma**0.5)
dd = normal(loc=rnorm, scale=1/rgamma**0.5)

[gn.Update(d) for d in dd]

#h = np.array(gn.surprise_history)
#dd = list(data1)
#dd.extend(data2)

a = 7.
b = 0.9
m = 8.
n = 300.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/rgamma**0.5)
dd = normal(loc=rnorm, scale=1/rgamma**0.5)

[gn.Update(d) for d in dd]
