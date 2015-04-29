from sampler import sampler
import numpy as np
import time
import os

X=np.load('Data/SimulatedData.npy')
N=X.shape[0]
D=X.shape[1]
sigmaX=1.
sigmaA=1.
alpha=1.
maxNew=4
niter=1000
BurnIn = 200

sampler(X, niter, BurnIn, sigmaX, sigmaA,alpha, N, D, maxNew)