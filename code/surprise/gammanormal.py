import numpy as np
from numpy import trace as tr
from numpy import log, pi
from numpy.linalg import inv, det
from scipy.special import psi, gamma, gammaln, digamma
import copy
import surprise
from surprise.base import SurpriseDistribution


class Gamma(SurpriseDistribution):
    def __init__(self):        
        super(Gamma, self).__init__()
        self.memory = 1
        self.parameters = {'alpha':1, 'beta':1}
    
    
    def _UpdateParams(self, data, data_mean):        
        #Calculate the updated gamma parameters and return them:
        old = self.parameters
        new = dict()
        new['alpha'] = self.memory * old['alpha'] + 0.5
        new['beta'] = self.memory * old['beta'] + (data-data_mean)**2/2
        return new
        
        
    def _SurpriseCalculation(self, old, new):
        a1 = old['alpha']
        b1 = old['beta']
        
        a2 = new['alpha']
        b2 = new['beta']
        
        return a1*log(b1) - a2*log(b2) - gammaln(a1) + gammaln(a2) + (a1-a2)*(digamma(a1)-log(b1)) + a1*(b2-b1)/b1


    def Expectation(self):
        return self.parameters['alpha'] / self.parameters['beta']
        
        
class Normal(SurpriseDistribution):
    def __init__(self):        
        super(Normal, self).__init__()
        self.memory = 1
        self.parameters = {'mu':1, 'tau':1}
    
    
    def _UpdateParams(self, data, data_precision):        
        #Calculate the updated gamma parameters and return them:
        old = self.parameters
        new = dict()
        new['tau'] = self.memory * old['tau'] + data_precision
        new['mu'] = (data*data_precision + self.memory*old['tau']*old['mu']) / (data_precision + self.memory*old['tau'])
        return new


    def _SurpriseCalculation(self, old, new):
        t1 = old['tau']
        m1 = old['mu']
        
        t2 = new['tau']
        m2 = new['mu']
        
        return 0.5*(log(t1) - log(t2) + (t2-t1)/t1 + (m1-m2)**2)


    def Expectation(self):
        return self.parameters['mu']
        
        
        
class GammaNormal(SurpriseDistribution):
    def __init__(self):        
        super(GammaNormal, self).__init__()
        self.memory = 1
        self.mean = Normal()
        self.precision = Gamma()
        self.parameters = {'mean':self.mean.Expectation(), 'precision':self.precision.Expectation()}

        
    def _UpdateParams(self, data):        
        #Calculate the updated hyperparameters
        old = self.parameters
        new = dict()
        
        self.mean.Update(data=data, data_precision=self.parameters['precision'])
        self.precision.Update(data=data, data_mean=self.parameters['mean'])
        
        new['mean'] = self.mean.Expectation()
        new['precision'] = self.precision.Expectation()
        return new


    def _SurpriseCalculation(self, old, new):       
        return [self.mean.surprise, self.precision.surprise]

        