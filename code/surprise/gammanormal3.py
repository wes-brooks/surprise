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
        self.window = 10
        self.n = 0
        self.parameters = {'a':1, 'b':1, 'p':1, 'm':1}
        self.sufficient = {'SS':0, 'ybar':0, 'n':0}
        self.lagged = {'SS':0, 'ybar':0, 'n':0}
        
        
    def _UpdateParams(self, data, **kwargs):        
        #Calculate the updated gamma parameters and return them:
        old = self.parameters
        new = dict()
        new['a'] = old['alpha'] + 0.5
        new['b'] = old['b'] + old['p']*(old['m']-data)**2. / (2*(1.+old['p']))
        new['p'] = old['p'] + 1.
        new['m'] = (data + old['m']*old['p'])/(1.+old['p'])
        
        prior_data = self.data_history[-(2*self.window):-self.window]
        update_data = self.data_history[-self.window:]
        
        old['m'] = mean(prior_data)
        
        
        return new
        
        
    def MomentEstimators(self, data):
        if len(data)<4:
            return {}
            
        data = np.array(data)    
        m1 = np.mean(data)
        m2 = np.mean(data**2)
        m3 = np.mean(data**3)
        m4 = np.mean(data**4)
        
        est = {}
        est['m'] = m1
        
        A = (m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4) / (m2-m1**2)**2
        est['a'] = (4*A-2) / (A-3)
        est['t'] = 1
        est['b'] = (m2-m1**2) * (est['a']-2) / 4.
        
        return est
        
        
    def _SurpriseCalculation(self, old, new):
        a1 = old['a']
        b1 = old['b']
        t1 = old['p']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        t2 = new['p']
        m2 = new['m']
        
        return (a1-a2)*(digamma(a1)-log(b1)) + a1*log(b1) - a2*log(b2) + 0.5*(log(t1/t2) + 1 - t2/t1 + a1*t1/b1*(m2-m1)**2) + gammaln(a2) - gammaln(a1) + (a1*(b2/b1 - 1))
        