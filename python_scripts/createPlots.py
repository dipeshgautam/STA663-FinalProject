import matplotlib
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import Image
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
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
plt.xlabel(r'(a) $K_+$')
plt.hist(chainK, bins = range(10), normed=True)
plt.subplot(122)
plt.hist(np.sum(Z,axis=1), bins = range(1,10))
plt.xlabel('(b) Features in an object')
plt.savefig('figures/kDistribution.png')

Z=Z[:,0:4]


def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

f=plt.figure(num=None, figsize=(12,6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot2grid((13,24),(0,0), colspan=6, rowspan=6)
plt.pcolormesh(A0[0,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(0,6), colspan=6, rowspan=6)
plt.pcolormesh(A0[1,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(0,12), colspan=6, rowspan=6)
plt.pcolormesh(A0[2,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(0,18), colspan=6, rowspan=6)
plt.pcolormesh(A0[3,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,0), colspan=6)
plt.pcolormesh(Z0[0,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,6), colspan=6)
plt.pcolormesh(Z0[1,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,12), colspan=6)
plt.pcolormesh(Z0[2,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,18), colspan=6)
plt.pcolormesh(Z0[3,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,0), colspan=6, rowspan=6)
plt.pcolormesh(X0[0,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,6), colspan=6, rowspan=6)
plt.pcolormesh(X0[1,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,12), colspan=6, rowspan=6)
plt.pcolormesh(X0[2,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,18), colspan=6, rowspan=6)
plt.pcolormesh(X0[3,:].reshape(6,6),cmap=plt.cm.gray)
make_ticklabels_invisible(f)
plt.savefig('figures/Original.png')

sigmaA=np.mean(chainSigmaA)
sigmaX=np.mean(chainSigmaX)
A_post=np.dot(np.dot(np.linalg.inv((np.dot(Z.T,Z)+(sigmaX**2/sigmaA**2)*np.eye(4))),Z.T),X0)

N=X0.shape[0]
D =X0.shape[1] 
Xpost=np.zeros((N,D))
for i in range(N):
    Xpost[i,:]=np.dot(Z[i,:],A_post[0:4,])


    
f=plt.figure(num=None, figsize=(12,6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot2grid((13,24),(0,0), colspan=6, rowspan=6)
plt.pcolormesh(A_post[0,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(0,6), colspan=6, rowspan=6)
plt.pcolormesh(A_post[1,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(0,12), colspan=6, rowspan=6)
plt.pcolormesh(A_post[2,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(0,18), colspan=6, rowspan=6)
plt.pcolormesh(A_post[3,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,0), colspan=6)
plt.pcolormesh(Z[0,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,6), colspan=6)
plt.pcolormesh(Z[1,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,12), colspan=6)
plt.pcolormesh(Z[2,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(6,18), colspan=6)
plt.pcolormesh(Z[3,:][np.newaxis,],cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,0), colspan=6, rowspan=6)
plt.pcolormesh(Xpost[0,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,6), colspan=6, rowspan=6)
plt.pcolormesh(Xpost[1,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,12), colspan=6, rowspan=6)
plt.pcolormesh(Xpost[2,:].reshape(6,6),cmap=plt.cm.gray)
plt.subplot2grid((13,24),(7,18), colspan=6, rowspan=6)
plt.pcolormesh(Xpost[3,:].reshape(6,6),cmap=plt.cm.gray)
make_ticklabels_invisible(f)    
    
plt.savefig('figures/Detected.png')


plt.figure(num=None, figsize = (12,6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(311)
plt.plot(chainSigmaX)
plt.ylabel(r'$\sigma_X$')
plt.subplot(312)
plt.plot(chainSigmaA)
plt.ylabel(r'$\sigma_A$')
plt.subplot(313)
plt.plot(chainAlpha)
plt.ylabel(r'$\alpha$')
plt.savefig('figures/Trace.png')