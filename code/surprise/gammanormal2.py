import numpy as np
from numpy import trace as tr
from numpy import log, pi
from numpy.linalg import inv, det
from scipy.special import psi, gamma, gammaln, digamma
import copy
import surprise


class GammaNormal(surprise.SurpriseDistribution):
    def __init__(self):        
        super(GammaNormal, self).__init__()
        self.memory = 1
        self.parameters = {'alpha':1, 'beta':1, 'mu':1, 'tau':1}
    
    
    def _UpdateParams(self, data):        
        #Calculate the updated gamma parameters and return them:
        old = self.parameters
        new = dict()
        new['alpha'] = self.memory * old['alpha'] + 0.5
        new['beta'] = self.memory*old['beta'] + self.memory*old['tau']/(1.+self.memory*old['tau']) * (old['mu']-data)**2.
        new['tau'] = self.memory * old['tau'] + 1.
        new['mu'] = (data + self.memory*old['mu']*old['tau'])/(1. + self.memory*old['tau'])
        return new
        
        
    def _SurpriseCalculation(self, old, new):
        print new
        print old
        a1 = old['alpha']
        b1 = old['beta']
        t1 = old['tau']
        m1 = old['mu']
        
        a2 = new['alpha']
        b2 = new['beta']
        t2 = new['tau']
        m2 = new['mu']
        
        return (a1-a2)*(digamma(a1)-log(b1)) + a1*log(b1) - a2*log(b2) + 0.5*(log(t1/t2) + 1 - t2/t1 + a1*t1/b1*(m2-m1)**2) + gammaln(a2) - gammaln(a1) + (a1*(b2/b1 - 1))
        #return new['alpha'] * log(old['beta']/new['beta']) + gammaln(new['alpha']) - gammaln(old['alpha']) + old['alpha']*(new['beta']-old['beta'])/old['beta'] + (old['alpha']-new['alpha']) * digamma(old['alpha'])