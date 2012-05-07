import numpy as np
from surprise import gammanormal3 as gammanormal

a=2.
b=1.
m=0.


cdom = np.loadtxt(fname="../data/2009_CDOM.csv", skiprows=0, dtype=float,unpack=False, delimiter=",")
pb = np.loadtxt(fname="../data/pheasantbranch-temperature.csv", skiprows=0, dtype=float,unpack=False, delimiter=",")
gills = np.loadtxt(fname="../data/GillsPondBrookTDS.csv", skiprows=0, dtype=float,unpack=False, delimiter=",")

    