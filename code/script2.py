import numpy as np
from surprise import gammanormal3
from numpy.random import poisson, gamma, normal

gnmock = gammanormal3.GammaNormal()
gnmock.window = 100

a = 20.
b = 2.
m = 10.
n = 600.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/rgamma**0.5)
dd = normal(loc=rnorm, scale=1/rgamma**0.5)

[gnmock.Update(d) for d in dd]


a = 2.
b = 2.
m = 10
n = 500.

rgamma = gamma(shape=a, scale=1./b, size=n)
rnorm = normal(loc=m, scale=1/rgamma**0.5)
dd = normal(loc=rnorm, scale=1/rgamma**0.5)

[gnmock.Update(d) for d in dd]
