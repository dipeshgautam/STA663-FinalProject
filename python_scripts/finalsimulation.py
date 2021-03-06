from sampler import sampler
import numpy as np
import time
import os
np.random.seed(1)


#Run the simulation with the best code
X=np.load('Data/SimulatedData.npy')
N=X.shape[0]
D=X.shape[1]
sigmaX=1.
sigmaA=1.
alpha=1.
maxNew=4
niter=1000
BurnIn = 0

sampler(X, niter, BurnIn, sigmaX, sigmaA,alpha, N, D, maxNew)