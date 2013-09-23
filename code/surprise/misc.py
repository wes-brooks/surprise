import numpy as np
from numpy import trace as tr
from numpy import log, pi, sqrt, exp
from numpy.linalg import inv, det
from scipy.special import psi, gamma as gammafn, gammaln, digamma

def f(a, b, m):
    y = (np.arange(20000.) - 10000)/100
    return gammafn(a+0.5) / (sqrt(4*pi*b) * gammafn(a)) * (1 + (y-m)**2/(4*b))**-(a+0.5)
    
    
def lik(a, b, m, y):
    return gammafn(a+0.5) / (sqrt(4*pi*b) * gammafn(a)) * (1 + (y-m)**2/(4*b))**-(a+0.5)