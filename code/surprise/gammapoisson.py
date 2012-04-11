import numpy as np
from numpy import trace as tr
from numpy import log, pi
from numpy.linalg import inv, det
from scipy.special import psi, gamma, gammaln, digamma
import copy
import surprise


class GammaPoisson(surprise.SurpriseDistribution):
    def __init__(self):        
        super(GammaPoisson, self).__init__()
        self.memory = 1
        self.parameters = {'alpha':1, 'beta':1}
    
    
    def _UpdateParams(self, data):        
        #Calculate the updated gamma parameters and return them:
        old = self.parameters
        new = dict()
        new['alpha'] = self.memory * old['alpha'] + data
        new['beta'] = self.memory * old['beta'] + 1
        return new
        
        
    def _SurpriseCalculation(self, old, new):
        print new
        print old
        a1 = old['alpha']
        b1 = old['beta']
        
        a2 = new['alpha']
        b2 = new['beta']
        
        return a1*log(b1) - a2*log(b2) - gammaln(a1) + gammaln(a2) + (a1-a2)*(digamma(a1)-log(b1)) + a1*(b2-b1)/b1
        #return new['alpha'] * log(old['beta']/new['beta']) + gammaln(new['alpha']) - gammaln(old['alpha']) + old['alpha']*(new['beta']-old['beta'])/old['beta'] + (old['alpha']-new['alpha']) * digamma(old['alpha'])