import numpy.testing as npt
import numpy as np
from __future__ import division
import math
import scipy.stats as stats
from functions import sampleIBP, calcInverse


sigmaA=1.
sigmaX=1.
alpha=1.
N=100
Kplus=4

np.random.seed(1)

Z=np.zeros((N,100))


for i in range(N):
    t=stats.poisson.rvs(alpha)
    if t>0:
        Z[i,0:t]=1

Z = Z[:,0:Kplus]

def test0():
    M1=np.linalg.inv(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    M2=np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus)
    npt.assert_almost_equal(np.dot(M1,M2),np.identity(Kplus))

def test1():
    (i,k,val) = (7,1,1)
    M1=calcInverse(Z,M,i,k,val)
    M2=np.linalg.inv(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    npt.assert_almost_equal(M1,M2, decimal =2)