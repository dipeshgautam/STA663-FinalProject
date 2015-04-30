from __future__ import division
import numpy.testing as npt
import numpy as np
import math
import scipy.stats as stats
from functions import sampleIBP, calcInverse, ll

X=np.load('Data/SimulatedData.npy')
chainSigmaX=np.load("Data/chainSigmaX.npy")
chainZ=np.load("Data/chainZ.npy")
sigmaA=1.
sigmaX=1.
alpha=1.
N=100
Kplus=4
D=36

np.random.seed(1)
Z, Kplus = sampleIBP(alpha, N)
#Z=np.zeros((N,100))


for i in range(N):
    t=stats.poisson.rvs(alpha)
    if t>0:
        Z[i,0:t]=1

Z = Z[:,0:Kplus]
M=np.linalg.inv(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))

#test of calcinverse
def testcalcInverse():
    (i,k,val) = (7,1,1)
    M1=calcInverse(Z,M,i,k,val)
    M2=np.linalg.inv(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    npt.assert_almost_equal(M1,M2, decimal =2)

#testing that likelihoods are positive    
def testll1():
    lik = np.exp(ll(X, Z, sigmaX, sigmaA, Kplus, D, N))
    assert lik >= 0
#Make sure that likelihood function gives zerodivision error when sigmaA is 0
def testll2():
    npt.assert_raises(ZeroDivisionError,ll, X, Z, sigmaX, 0, Kplus, D, N)

#Make sure likelihood gives nan if sigmaA is negative
def testll3():
    assert math.isnan(ll(X, Z, sigmaX, -.5, Kplus, D, N))==True

#Make sure likelihood gives nan if sigmaX is negative
def testll4():
    assert math.isnan(ll(X, Z, -0.5, sigmaA, Kplus, D, N))==True    

#test of convergence of code
def testconv1():
    assert (np.abs(np.mean(chainSigmaX[200:])-0.5))<=.05
    
#test that each object has at least one feature as we asserted that while simulating data
Zfinal = chainZ[-1,:,0:4]
def testconv2():
    assert np.sum(Zfinal,axis=1).all()>=1