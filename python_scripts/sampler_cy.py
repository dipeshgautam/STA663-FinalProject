import numpy as np
import scipy as sp
import math
import Cython_functions as func

def sampler_cy(X, niter, BURN_IN, sigmaX, sigmaA,alpha, N, D, maxNew):
    """Implementation of MCMC which includes Gibbs sampling as well as MH steps."""
    HN = 0.
    for i in range(1,N+1):
        HN += 1./i

    SampleSize=niter-BurnIn
    
    #Even though we can have infinite features, for this problem, for efficiency in storing, set it to 20 as we never get ther
    K_inf=20
    
    #initialize matrices to store the samples

    chainZ=np.zeros((SampleSize,N,K_inf))
    chainK=np.zeros((SampleSize,1))
    chainSigma_X=np.zeros((SampleSize,1))
    chainSigma_A=np.zeros((SampleSize,1))
    chainAlpha=np.zeros((SampleSize,1))
    np.random.seed(1)
    Z, Kplus = func.sampleIBP(alpha, N)
    s_counter=0

    for j in range(niter):
        #print("iteration:",j ,  "Kplus:",Kplus,  "shape of Z", Z.shape, "alpha:", alpha, "sigmaX", sigmaX)
        #update z
        if((j+1)>BURN_IN):
            chain_Z[s_counter,:,0:Kplus]=Z
            chain_K[s_counter]=Kplus
            chain_sigma_X[s_counter]=sigmaX
            chain_sigma_A[s_counter]=sigmaA
            chain_alpha[s_counter]=alpha
            s_counter=s_counter+1

        for i in range(N):
            for k in range(Kplus):
                #print k
                if k>=Kplus:
                    break     
                #Removing the singular features, i.e. the ones that have 1 for the current object only.
                if Z[i,k] > 0:
                    if (np.sum(Z[:,k])- 1) <=0:
                        #Z[i,k] = 0
                        Z[:,k:(Kplus-1)] = Z[:,(k+1):Kplus] #shift everything one column to the left
                        Kplus = Kplus-1
                        Z = Z[:,0:Kplus] # remove the last column as it is redundent
                        continue #We're no longer looking at this feature, so move to another one               

                P = np.zeros(2)
                #set Z[i,k] = 0 and calculate posterior probability
                Z[i,k] = 0
                P[0] = func.ll(X, Z, sigmaX, sigmaA, Kplus, D, N) + np.log(N-np.sum(Z[:,k])) - np.log(N)

                #set Z[i,k] = 1 and calculate posterior probability
                Z[i,k] = 1
                P[1] = func.ll(X, Z,sigmaX, sigmaA, Kplus, D, N)  + np.log(np.sum(Z[:,k])- 1) - np.log(N)

                P = np.exp(P - max(P))
                U = np.random.uniform(0,1)
                if U<(P[1]/(np.sum(P))):
                    Z[i,k] = 1
                else:
                    Z[i,k] = 0   


            #Sample number of new features
            prob = np.zeros(maxNew)
            alphaN = alpha/N
            for kNew in range(maxNew): # max new features is 3
                Z_temp = Z
                if kNew>0:
                    addCols = np.zeros((N,kNew))
                    addCols[i,:] = 1
                    Z_temp = np.hstack((Z_temp, addCols))

                #Calculate the probability of kNew new features for object i
                pois = kNew*np.log(alphaN) - alphaN - np.log(math.factorial(kNew))
                lik = func.ll(X = X, Z = Z_temp, sigmaX = sigmaX, sigmaA = sigmaA, K=(Kplus+kNew), D= D, N= N)
                prob[kNew] = pois + lik

            #normalize prob and select the most likely number of new features
            prob = np.exp(prob - max(prob))
            prob = prob/sum(prob)

            U = np.random.uniform(0,1,1)
            p = 0
            kNew=0
            for new in range(maxNew):
                p = p+prob[new]
                if U<p:
                    kNew = new
                    break

            #Add kNew new columns to Z and set the values at ith row to 1 for all of them
            if kNew>0:
                addCols = np.zeros((N,kNew))
                addCols[i,:] = 1
                Z = np.hstack((Z, addCols))
            Kplus = Kplus + kNew 

        llCurrent = func.ll(X, Z, sigmaX, sigmaA, Kplus, D, N )
        #update sigmaX
        if np.random.uniform(0,1) < .5:
            sigmaX_new = sigmaX - np.random.uniform(0,1)/20
        else:
            sigmaX_new = sigmaX + np.random.uniform(0,1)/20
        llNew = func.ll(X, Z, sigmaX_new, sigmaA, Kplus, D, N)

        arX = np.exp(min(0,llNew-llCurrent))
        U = np.random.uniform(0,1)
        if U < arX:
            sigmaX = sigmaX_new

        #update sigma_A
        if np.random.uniform(0,1) < .5:
            sigmaA_new = sigmaA - np.random.uniform(0,1)/20
        else:
            sigmaA_new = sigmaA + np.random.uniform(0,1)/20
            
        llNew = func.ll(X, Z, sigmaX, sigmaA_new, Kplus, D, N)

        arA = np.exp(min(0,llNew-llCurrent))

        U = np.random.uniform(0,1)
        if U < arA:
            sigmaA = sigmaA_new

        alpha = np.random.gamma(1+Kplus, 1/(1+HN))