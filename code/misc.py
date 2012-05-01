import numpy as np
from numpy import trace as tr
from numpy import log, pi, sqrt, exp
from numpy.linalg import inv, det
from scipy.special import gamma as gammafn
import copy

a=2.
b=1.
m=0.


def f(a, b, m):
    y = (np.arange(20000.) - 10000) / 100
    return gammafn((2*a+1)/2) / gammafn(a) / sqrt(pi*4*b) * (1+(y-m)**2 / (4*b))**-((2*a+1)/2)
    