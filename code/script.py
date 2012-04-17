import numpy as np
from surprise import gammapoisson
from numpy.random import poisson, gamma, normal


gp = gammapoisson.GammaPoisson()
gp.memory = 0.7


a = 50.
b = 5.
n = 100.

rgamma = gamma(shape=a, scale=1./b, size=n)
rpois = poisson(lam=rgamma)

[gp.Update(l) for l in rpois]

rp = gamma(shape=a, scale=1./b, size=n)
mu = normal(loc=p, scale=1/(t*rp))
X = normal(mu, rp)