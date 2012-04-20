import numpy as np
from surprise import gammapoisson, gammanormal
from numpy.random import poisson, gamma, normal


gp = gammapoisson.GammaPoisson()
gp.memory = 0.96

a = 50.
b = 5.
n = 100.

rgamma = gamma(shape=a, scale=1./b, size=n)
rpois = poisson(lam=rgamma)

[gp.Update(l) for l in rpois]


a = 25.
b = 5.
n = 100.

rgamma = gamma(shape=a, scale=1./b, size=n)
rpois = poisson(lam=rgamma)

[gp.Update(l) for l in rpois]




gn = gammanormal.GammaNormal()
gn.memory = 0.95

a = 50.
b = 50.
t = 5.
m = 50.
n = 100.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/(rgamma*t)**0.5)
data = normal(loc=rnorm, scale=1/rgamma**0.5)

[gn.Update(d) for d in data]


a = 50.
b = 5.
t = 10.
m = 50.
n = 100.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/(rgamma*t)**0.5)
data = normal(loc=rnorm, scale=1/rgamma**0.5)

[gn.Update(d) for d in data]
