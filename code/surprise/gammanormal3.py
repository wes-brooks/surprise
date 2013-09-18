import numpy as np
from numpy import trace as tr
from numpy import log, pi, sqrt, exp
from numpy.linalg import inv, det
from scipy.special import psi, gamma as gammafn, gammaln, digamma
import copy

import misc
import surprise
from surprise.base import SurpriseDistribution
       
default_parameters = {'a':1, 'b':1, 'm':1}
        
class GammaNormal(SurpriseDistribution):
    def __init__(self):        
        super(GammaNormal, self).__init__()
        self.window = 96
        self.n = 0
        self.parameters = default_parameters
        self.sufficient = {'SS':0, 'ybar':0, 'n':0}
        self.lagged = {'SS':0, 'ybar':0, 'n':0}
        
        
    def _UpdateParams(self, data, **kwargs):        
        #Calculate the updated gamma parameters and return them:
        if(len(self.data_history) <= self.window): 
            return {'old':default_parameters, 'new':default_parameters}

        prior_data = self.data_history[-(2*self.window):-self.window]
        #prior_data = self.data_history[:-self.window]
        update_data = self.data_history[-self.window:]
        
        old = self.MomentEstimators(prior_data)
        new = self.MomentEstimators(update_data)
        
        #old['m'] = mean(prior_data)
        return {'old':old, 'new':new}
        
        
    def MomentEstimators(self, data):
        if len(data)<4:
            return default_parameters
            
        data = np.array(data)    
        m1 = np.mean(data)
        m2 = np.mean(data**2)
        m3 = np.mean(data**3)
        m4 = np.mean(data**4)
                
        cm4 = m4 - 4*m1*m3 + 6*m2*m1**2 - 3*m1**4
        cm2 = (m2-m1**2)
        est = {}
        
        est['m'] = m1        
        est['a'] = 2 + 3*cm2**2/(4*cm4)
        est['b'] = (est['a']-1) * (m2-m1**2) / 2
        
        print est
        
        return est
        
        
    def h(self, y, t=0):
        old = self.parameters['old']
        new = self.parameters['new']
        
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2
        
        return -0.5*(t*A2*log(1 + (y-m2)**2/B2) + A1*log(1 + (y-m1)**2/B1)) 
        
        
    def hp(self, y, t=0):
        old = self.parameters['old']
        new = self.parameters['new']
        
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2
        
        return -(t*A2*(y-m2)/(B2+(y-m2)**2) + A1*(y-m1)/(B1+(y-m1)**2))
        
        
    def hpp(self, y, t=0):
        old = self.parameters['old']
        new = self.parameters['new']
        
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2
        
        return -((t*A2*(B2+(y-m2)**2) - 2*t*A2*(y-m2)**2)/(B2+(y-m2)**2)**2 + (A1*(B1+(y-m1)**2) - 2*A1*(y-m1)**2)/(B1+(y-m1)**2)**2)
        
        
    def hppp(self, y, t=0):
        old = self.parameters['old']
        new = self.parameters['new']
        
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2
        
        return -(-2*t*A2*(y-m2)/(B2+(y-m2)**2)**4 - (4*t*A2*(y-m2)*(B2+(y-m2)**2)**2 - 2*(B2+(y-m2)**2) * 2*(y-m2) * 2*t*A2*(y-m2)**2)/(B2+(y-m2)**2)**4 - \
                2*t*A1*(y-m1)/(B1+(y-m1)**2)**4 - (4*t*A1*(y-m1)*(B1+(y-m1)**2)**2 - 2*(B1+(y-m1)**2) * 2*(y-m1) * 2*t*A1*(y-m1)**2)/(B1+(y-m1)**2)**4)
        
    def r(self, t=0):
        old = self.parameters['old']
        new = self.parameters['new']
                
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2

        #Coefficients of the cubic function, evaluated at t
        a = A1 + t*A2
        b = -A1*(m1+2*m2) - t*A2*(m2+2*m1)
        c = A1*(m2**2 + B2 + 2*m1*m2) + t*A2*(m1**2 + B1 + 2*m1*m2)
        d = -A1*m1*(m2**2 + B2) - t*A2*m2*(m1**2 + B1)
        
        delta = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
        
        r = -1/(3.*a) * (b + \
                (0.5*(2*b**3 - 9*a*b*c + 27*a**2*d + sqrt(-27*a**2*delta)))**(1./3) - \
                (-0.5*(2*b**3 - 9*a*b*c + 27*a**2*d - sqrt(-27*a**2*delta)))**(1./3))
                
        return r
 
 
    def rp(self, t=0):
        old = self.parameters['old']
        new = self.parameters['new']
            
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2

        #Coefficients of the cubic function, evaluated at t=0
        a = A1 + t*A2
        b = -A1*(m1+2*m2) - t*A2*(m2+2*m1)
        c = A1*(m2**2 + B2 + 2*m1*m2) + t*A2*(m1**2 + B1 + 2*m1*m2)
        d = -A1*m1*(m2**2 + B2) - t*A2*m2*(m1**2 + B1)
        
        delta = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
        
        #Derivatives of coefficients of the cubic function, evaluated at t-0:
        ap = A2
        bp = -A2*(2*m1 + m2)
        cp = A2*(m1**2 + B1 + 2*m1*m2)
        dp = -A2*m2*(m1**2 + B1)
        
        deltap = 18*(ap*b*c*d + a*bp*c*d + a*b*cp*d + a*b*c*dp) - \
                    4*(3*b**2*bp*d + b**3*dp) + \
                    (2*b*bp*c**2 + 2*b**2*c*cp) - \
                    4*(ap*c**3 + a*3*c**2*cp) - \
                    27*(2*a*ap*d**2 + 2*a**2*d*dp)
                    
        rp = ap/(3.*a**2) * (b + (0.5*(2*b**3 - 9*a*b*c + 27*a**2*d + sqrt(-27*a**2*delta)))**(1./3) - (-0.5*(2*b**3 - 9*a*b*c + 27*a**2*d - sqrt(-27*a**2*delta)))**(1./3)) - \
                1/(3.*a) * (bp + 1./3 * (0.5*(2*b**3 - 9*a*b*c + 27*a**2*d + sqrt(-27*a**2*delta)))**(-2./3) * 0.5*(6*b**2*bp - 9*(ap*b*c + a*bp*c + a*b*cp) + \
                27*(2*a*ap*d + a**2*dp) + 0.5*(-27*a**2*delta)**(-0.5) * -27*(2*a*ap*delta + a**2*deltap)) - \
                1./3 * (-0.5*(2*b**3 - 9*a*b*c + 27*a**2*d - sqrt(-27*a**2*delta)))**(-2./3) * 0.5*(6*b**2*bp - 9*(ap*b*c + a*bp*c + a*b*cp) + \
                27*(2*a*ap*d + a**2*dp) - 0.5*(-27*a**2*delta)**(-0.5) * -27*(2*a*ap*delta + a**2*deltap)))
                
        return rp        
        
        
    def Etop(self):
        old = self.parameters['old']
        new = self.parameters['new']
            
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']
        
        A1 = 2*a1 + 1
        A2 = 2*a2 + 1
        
        B1 = 4*b1
        B2 = 4*b2    
        
        h = self.h
        hp = self.hp
        hpp = self.hpp
        hppp = self.hppp
        
        r = self.r
        rp = self.rp
        
        y = r(0)
        
        return sqrt(2*pi) * exp(h(y)) * (hp(y)*rp(0)*(-hpp(y))**(-0.5) + \
                        (0.5)*(-hpp(y))**(-1.5)*hppp(y)*rp(0))
        
        
    def _SurpriseCalculation(self, ignore_me, params):
        old = params['old']
        new = params['new']
                    
        a1 = old['a']
        b1 = old['b']
        m1 = old['m']
        
        a2 = new['a']
        b2 = new['b']
        m2 = new['m']       
        
        #ElogY1 = gammaln(a1+0.5) - gammaln(a1) - 0.5*log(4*pi*b1) - 0.5
        #ElogY1 = gammaln(a1+0.5) - gammaln(a1) - 0.5*log(4*pi*b1) + (a1+0.5)*(digamma(a1)-digamma(a1+0.5))
        #ElogY2 = gammaln(a2+0.5) - gammaln(a2) - 0.5*log(4*pi*b2) + self.Etop() / sqrt(4*pi*b1/(a1+0.5))
        
        #print ElogY1
        #print ElogY2
        
        #return ElogY1 - ElogY2
        
        return sum(log(misc.f(a1, b1, m1)/misc.f(a2, b2, m2)) * misc.f(a1, b1, m1))/100