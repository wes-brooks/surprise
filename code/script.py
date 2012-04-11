import numpy as np
from surprise import gammapoisson
from scipy.stats import poisson, gamma


gp = gammapoisson.GammaPoisson()
gp.memory = 0.7


a = 50.
b = 5.
n = 100.

rgamma = gamma.rvs(a=a, scale=1./b, size=n)
rpois = poisson.rvs(mu=rgamma)

[gp.Update(l) for l in rpois]