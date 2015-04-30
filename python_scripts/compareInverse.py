import time
import numpy as np
import pandas as pd
from functions import calcInverse, sampleIBP
import os
np.random.seed(1)

X=np.load('Data/SimulatedData.npy')
N=X.shape[0]
D=X.shape[1]
sigmaX=1.
sigmaA=1.
alpha=1.


#calculate and time the inversion method described by Griffiths and Ghahramani(2005)
i=10
k=3

Z,K = sampleIBP(alpha,N)

M = np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))
Z[i,k] = 1
val = 0
loops = 1000
tcalcInv=np.zeros(loops)
for l in range(loops):
    t0=time.time()
    calcInverse(Z,M,i,k,val)
    t1=time.time()
    tcalcInv[l]=t1-t0
mtcalcInv= round(np.mean(tcalcInv),7)



#time np.linalg.inv
tlinalgInv=np.zeros(loops)
for l in range(loops):
    t0=time.time()
    np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))
    t1=time.time()
    tlinalgInv[l]=t1-t0
mtlinalgInv= round(np.mean(tlinalgInv),7)


times = np.array((mtlinalgInv,mtcalcInv))


#save result to latex table to be used in report
columns = ['Time']
index = ['linalg.inverse','calcInverse']

if not os.path.exists('latex_tables'):
    os.makedirs('latex_tables')
df = pd.DataFrame(times,columns=columns,index=index)
tab = df.to_latex()
text_file = open("latex_tables/inverseMethods.tex", "w")
text_file.write(tab)
text_file.close()
