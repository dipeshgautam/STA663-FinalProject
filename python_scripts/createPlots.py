import matplotlib
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import Image
import matplotlib.cm as cm
import os

if not os.path.exists('figures'):
    os.makedirs('figures')


chainZ=np.load("Data/chainZ.npy")
chainK=np.load("Data/chainK.npy")
chainSigmaX=np.load("Data/chainSigmaX.npy")
chainSigmaA=np.load("Data/chainSigmaA.npy")
chainAlpha=np.load("Data/chainAlpha.npy")
Z = chainZ[-1,:,:]

X0 = np.load("Data/SimulatedData.npy")
Z0 = np.load("Data/ZOriginal.npy")
A0 = np.load("Data/AOriginal.npy")

plt.figure(num=None, figsize=(12,3), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(121)
plt.hist(chainK, bins = range(10), normed=True)
plt.subplot(122)
plt.hist(np.sum(Z,axis=1), bins = range(1,10))
plt.savefig('figures/kDistribution.png')

Z=Z[:,0:4]
plt.figure(num=None, figsize=(12,6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(241)
plt.pcolormesh(A0[0,:].reshape(6,6),cmap=plt.cm.gray)     
plt.subplot(242)
plt.pcolormesh(A0[1,:].reshape(6,6),cmap=plt.cm.gray)  
plt.subplot(243)
plt.pcolormesh(A0[2,:].reshape(6,6),cmap=plt.cm.gray)  
plt.subplot(244)
plt.pcolormesh(A0[3,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot(245)
plt.pcolormesh(X0[0,:].reshape(6,6),cmap=plt.cm.gray)     
plt.subplot(246)
plt.pcolormesh(X0[1,:].reshape(6,6),cmap=plt.cm.gray)  
plt.subplot(247)
plt.pcolormesh(X0[2,:].reshape(6,6),cmap=plt.cm.gray)  
plt.subplot(248)
plt.pcolormesh(X0[3,:].reshape(6,6),cmap=plt.cm.gray)
plt.savefig('figures/Original.png')

sigmaA=np.mean(chainSigmaA)
sigmaX=np.mean(chainSigmaX)
A_post=np.dot(np.dot(np.linalg.inv((np.dot(Z.T,Z)+(sigmaX**2/sigmaA**2)*np.eye(4))),Z.T),X0)

N=X0.shape[0]
D =X0.shape[1] 
Xpost=np.zeros((N,D))
for i in range(N):
    Xpost[i,:]=np.dot(Z[i,:],A_post[0:4,])


plt.figure(num=None, figsize=(12,6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(241)
plt.pcolormesh(A_post[0,:].reshape(6,6),cmap=plt.cm.gray)     
plt.subplot(242)
plt.pcolormesh(A_post[1,:].reshape(6,6),cmap=plt.cm.gray)  
plt.subplot(243)
plt.pcolormesh(A_post[2,:].reshape(6,6),cmap=plt.cm.gray)  
plt.subplot(244)
plt.pcolormesh(A_post[3,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot(245)
plt.pcolormesh(Xpost[0,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot(246)
plt.pcolormesh(Xpost[1,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot(247)
plt.pcolormesh(Xpost[2,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot(248)
plt.pcolormesh(Xpost[3,:].reshape(6,6),cmap=plt.cm.gray)
plt.savefig('figures/Detected.png')


plt.figure(num=None, figsize = (12,6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(221)
plt.plot(chainSigmaX)
plt.subplot(222)
plt.plot(chainSigmaA)
plt.subplot(223)
plt.plot(chainAlpha)
plt.savefig('figures/Trace.png')


if not os.path.exists('latex_tables'):
    os.makedirs('latex_tables')

index=['1st image','2nd image','3rd image','4th image']
columns=['F1','F2','F3','F4']


df = pd.DataFrame(np.hstack((np.vstack([Z0[0,0],Z0[1,0],Z0[2,0],Z0[3,0]]),np.vstack([Z0[0,1],Z0[1,1],Z0[2,1],Z0[3,1]]) \
                             ,np.vstack([Z0[0,2],Z0[1,2],Z0[2,2],Z0[3,2]]),np.vstack([Z0[0,3],Z0[1,3],Z0[2,3],Z0[3,3]]))) \
                  ,index=index,columns=columns)
tab = df.to_latex()
text_file = open("latex_tables/featuresDetected.tex", "w")
text_file.write(tab)
text_file.close()
