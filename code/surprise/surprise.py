import numpy as np
from numpy import trace as tr
from numpy import log, pi
from numpy.linalg import inv, det
from scipy.special import psi, gammaln, digamma, gamma as gammafn
import copy


class SurpriseDistribution(object):
    def __init__(self):
        self.surprise = list()
        self.parameter_history = list()
        self.surprise_history = list()
        self.data_history = list()
        
    def _UpdateParameters(self, **kwargs):
        raise NotImplementedError
    
    def _SurpriseCalculation(self, **kwargs):
        raise NotImplementedError

    def ArchiveParameters(self, parameters):
        self.parameter_history.append(parameters)
        
    def ArchiveSurprise(self, surprise):
        self.surprise_history.append(surprise)
        
    def ArchiveData(self, data):
        self.data_history.append(data)
        
    def Update(self, data, **kwargs):
        #Save the old parameters and generate the new ones:
        old = copy.copy(self.parameters)
        self.parameters = self._UpdateParams(data, **kwargs)
    
        #Calculate the surprise in the new data and archive it along with the latest parameters.
        self.surprise = surprise = self.Surprise(old, self.parameters)
        self.ArchiveParameters(self.parameters)
        self.ArchiveSurprise(surprise)
        self.ArchiveData(data)
    
    def Surprise(self, old, new):
        surprise = self._SurpriseCalculation(old, new)
        return surprise
        
    