import numpy as np
from numpy import trace as tr
from numpy import log, pi
from numpy.linalg import inv, det
from scipy.special import psi, gamma, gammaln, digamma
import copy
import surprise

        
class NormalWishart(surprise.SurpriseDistribution):
    def __init__(self):
        super(NormalWishart, self).__init__()
        self.memory = 1
        self.parameters = {'t':1, 'm':1, 'alpha':1, 'tau':1}
        
    
    def _UpdateParams(self, data, old_parameters):
        #Calculate the updated gamma parameters and return them:
        old = old_parameters
        new = dict()
        new['alpha'] = self.memory * old['alpha'] + log(old['beta']) - digamma(self.memory*old['alpha']) + log(data)
        new['beta'] = new['alpha'] * (self.memory * old['beta'] + 1)
        return new   
       
       
    def _Z(alpha, tau):
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
    
    
    def _SurpriseCalculation(self, old, new):
        return new['alpha'] * log(old['beta']/new['beta']) + gammaln(new['alpha']) - gammaln(old['alpha']) + old['alpha']*(new['beta']-old['beta'])/old['beta'] + (old['alpha']-new['alpha']) * digamma(old['alpha'])

        
        

        




    

    