import numpy as np
from numpy import trace as tr
from numpy import log, pi
from numpy.linalg import inv, det
from scipy.special import psi, gamma, gammaln, digamma
import copy


class Gamma(object):
    def __init__(self):
        self.surprise = np.array()
        self.memory = 1
        self.parameters = [{'alpha':1, 'beta':1}]
    
    def _UpdateParams(self, data):
        #Get the current gamma parameters from the end of the list:
        old = self.parameters[-1]
        
        #Calculate the updated gamma parameters and return them:
        new = dict()
        new['alpha'] = self.memory * old['alpha'] + log(old['beta']) - digamma(self.memory*old['alpha']) + log(data)
        new['beta'] = new['alpha'] * (self.memory * old['beta'] + 1) / 
        return new
        
    
    def Update(self, data):
        try:
            #Check whether we got a list of data, but not a string:
            assert hasattr(data, '__iter__')
            assert not isinstance(data, str)
            new_parameters = [self._UpdateParams(x) for x in data]
            surprise = self.Surprise(new_parameters)
        
        except AssertionError: #This means we got one observation, not a list.    
            new_parameters = self._UpdateParams(data)
    
    def Surprise(self, new):
            try:
            #Check whether we got a list of data, but not a string:
            assert hasattr(data, '__iter__')
            assert not isinstance(data, str)
            new_parameters = [self._UpdateParams(x) for x in data]
            surprise = self.Surprise(new_parameters)
        
        except AssertionError: #This means we got one observation, not a list.    
            self.parameters.extend(self._UpdateParams(data))
    
    
    def _SurpriseCalculation(self, old, new):
        return new['alpha'] * log(old['beta']/new['beta']) + gammaln(new['alpha']) - gammaln(old['alpha']) + old['alpha']*(new['beta']-old['beta'])/old['beta'] + (old['alpha']-new['alpha']) * digamma(old['alpha'])
        

def Z(alpha, tau):
    d = tau.shape[1]
    i = np.arange(d)
    return pi**(d*(d-1)/4) * det(tau/2)**(-alpha/2) * np.prod(gamma((a+1-i)/2))

def Wishart(old, new):
    #First get the expectation of the log of the determinant of rho:
    d = old['tau'].shape[1]
    i = np.arange(d)
    ER = -log(det(old['tau']) / 2) + sum(psi((old['alpha'] - i + 1)/2))
    
    #Now compute the overall KL divergence:
    return (old['alpha']-new['alpha']) / 2 * ER - old['alpha'] * d / 2 + old['alpha'] / 2. * tr(new['tau'] * inv(old['tau'])) + log(Z(new['alpha'], new['tau'])) - log(Z(old['alpha'], old['tau']))

def NormalWishart(old, new):
    d = old['tau'].shape[1]
    return Wishart(old, new) + 0.5 * (d * log(old['t'] / new['t']) + d * new['t'] / old['t'] - d + new['t'] * (old['m']-new['m']).T * old['alpha'] * inv(old['tau']) * (old['m']-new['m']))
    
    

    