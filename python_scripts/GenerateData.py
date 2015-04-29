from __future__ import division
import numpy as np
import math
import os

np.random.seed(1)

N = 100 #number of objects
K = 4 #true number of features
D = 36 # dimension of feature


sigmaX0 = .5;
A = np.array((0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)).reshape(4, D)


I = (sigmaX0)*np.identity(D)
Z0 = np.zeros((N, K))
X = np.zeros((N, D))
for i in range(N):
    Z0[i,:] = (np.random.uniform(0,1,K) > .5).astype(int)
    while (np.sum(Z0[i,:]) == 0):
        Z0[i,:] = (np.random.uniform(0,1,K) > .5).astype(int)
    X[i,:] = np.random.normal(0,1, (1,D)).dot(I)+Z0[i,:].dot(A)
    
if not os.path.exists('Data'):
    os.makedirs('Data')
np.save("Data/SimulatedData", X)
np.save("Data/ZOriginal", Z0)
np.save("Data/AOriginal", A)