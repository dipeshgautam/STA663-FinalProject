#from __future__ import division
#np.random.seed(1)
#Sample prior
def sampleIBP(alpha, N):
    import numpy as np
    import math
    np.random.seed(1)
    result = np.zeros((N, 1000))
    t = np.random.poisson(alpha)
    if t>0:
        result[0,0:t] = np.ones(t)
    Kplus = t
    for i in range(1,N):
        for j in range(Kplus):
            p = np.sum(result[0:i,j])/(i+1)
            if np.random.uniform(0,1) < p:
                result[i,j] = 1
        t = np.random.poisson(alpha/(i+1))
        if t>0:
            result[i,Kplus:Kplus+t] = np.ones(t)
            Kplus = Kplus+t
    result = result[:,0:Kplus]
    return np.array((result, Kplus))

#Rank one update inverse calculation
def calcInverse(Z,M,i,k,val):
    import numpy as np
    import math
    M_i = M - M.dot(Z[i,:].T.dot(Z[i,:].dot(M)))/(Z[i,:].dot(M.dot(Z[i,:].T))-1)
    Z[i,k] = val
    M = M_i - M_i.dot(Z[i,:].T.dot(Z[i,:].dot(M_i)))/(Z[i,:].dot(M_i.dot(Z[i,:].T))+1)
    Inv = M
    return Inv

#Original likelihood function
def ll0(X, Z, sigmaX, sigmaA, K, D, N):
    import numpy as np
    import math
    return (-1)*np.log(2*np.pi)*N*D*.5 - np.log(sigmaX)*(N-K)*D - np.log(sigmaA)*K*D \
- .5*D*np.log(np.linalg.det(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))) \
-.5/(sigmaX**2)*np.trace((X.T.dot( np.identity(N) \
                                  -Z.dot(np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K)).dot(Z.T)))).dot(X))

#Improved Likelihood function
def ll(X, Z, sigmaX, sigmaA, K, D, N):
    import numpy as np
    import math
    M = Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K)
    return (-1)*np.log(2*np.pi)*N*D*.5 - np.log(sigmaX)*(N-K)*D - np.log(sigmaA)*K*D - .5*D*np.log(np.linalg.det(M)) \
        -.5/(sigmaX**2)*np.trace( (X.T.dot( np.identity(N)-Z.dot(np.linalg.inv(M).dot(Z.T)) )).dot(X) )