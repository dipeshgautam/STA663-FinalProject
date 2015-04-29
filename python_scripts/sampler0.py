from __future__ import division
import numpy as np
import math
from functions import sampleIBP
from functions import ll0 as ll

def sampler0(X, niter, BurnIn, sigmaX, sigmaA,alpha, N, D, maxNew):
    np.random.seed(1)
    HN = 0.
    for i in range(1,N+1):
        HN += 1./i

    SampleSize=niter-BurnIn

    K_inf=20

    chainZ=np.zeros((SampleSize,N,K_inf))
    chainK=np.zeros((SampleSize,1))
    chainSigmaX=np.zeros((SampleSize,1))
    chainSigmaA=np.zeros((SampleSize,1))
    chainAlpha=np.zeros((SampleSize,1))
    Z, Kplus = sampleIBP(alpha, N)
    s_counter=0

    for j in range(niter):
        if((j+1)>BurnIn):
            chainZ[s_counter,:,0:Kplus]=Z
            chainK[s_counter]=Kplus
            chainSigmaX[s_counter]=sigmaX
            chainSigmaA[s_counter]=sigmaA
            chainAlpha[s_counter]=alpha
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
                P[0] = ll(X, Z, sigmaX, sigmaA, Kplus, D, N) + np.log(N-np.sum(Z[:,k])) - np.log(N)

                #set Z[i,k] = 1 and calculate posterior probability
                Z[i,k] = 1
                P[1] = ll(X, Z,sigmaX, sigmaA, Kplus, D, N)  + np.log(np.sum(Z[:,k])- 1) - np.log(N)

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

                pois = kNew*np.log(alphaN) - alphaN - np.log(math.factorial(kNew))
                lik = ll(X = X, Z = Z_temp, sigmaX = sigmaX, sigmaA = sigmaA, K=(Kplus+kNew), D= D, N= N)
                prob[kNew] = pois + lik

            #normalize prob
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

        llCurrent = ll(X, Z, sigmaX, sigmaA, Kplus, D, N )
        #update sigmaX
        if np.random.uniform(0,1) < .5:
            sigmaX_new = sigmaX - np.random.uniform(0,1)/20
        else:
            sigmaX_new = sigmaX + np.random.uniform(0,1)/20
        llNew = ll(X, Z, sigmaX_new, sigmaA, Kplus, D, N)

        arX = np.exp(min(0,llNew-llCurrent))
        U = np.random.uniform(0,1)
        if U < arX:
            sigmaX = sigmaX_new

        if np.random.uniform(0,1) < .5:
            sigmaA_new = sigmaA - np.random.uniform(0,1)/20
        else:
            sigmaA_new = sigmaA + np.random.uniform(0,1)/20

        llNew = ll(X, Z, sigmaX, sigmaA_new, Kplus, D, N)

        arA = np.exp(min(0,llNew-llCurrent))
        U = np.random.uniform(0,1)
        if U < arA:
            sigmaA = sigmaA_new

        alpha = np.random.gamma(1+Kplus, 1/(1+HN))