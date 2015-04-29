import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def ll(X, Z, sigmaX, sigmaA, K, D, N):
    #M = Z[:,0:K].T.dot(Z[:,0:K])+sigmaX**2/sigmaA**2*np.identity(K)
    M = Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K)
    return (-1)*np.log(2*np.pi)*N*D*.5 - np.log(sigmaX)*(N-K)*D - np.log(sigmaA)*K*D - .5*D*np.log(np.linalg.det(M)) \
        -.5/(sigmaX**2)*np.trace( (X.T.dot( np.identity(N)-Z.dot(np.linalg.inv(M).dot(Z.T)) )).dot(X) )

np.random.seed(1)
def sampleIBP(alpha, N):
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