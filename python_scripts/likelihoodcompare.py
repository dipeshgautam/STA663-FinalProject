from functions import sampleIBP, ll, ll0
import pandas as pd
import numpy as np
import time
import os
np.random.seed(1)

X=np.load('Data/SimulatedData.npy')
N=X.shape[0]
D=X.shape[1]
sigmaX=1.
sigmaA=1.
alpha=1.

Z,K = sampleIBP(alpha,N)

loops = 1000
tll0=np.zeros(loops)
for l in range(loops):
    t0=time.time()
    ll0(X, Z, sigmaX, sigmaA, K, D, N)
    t1=time.time()
    tll0[l]=t1-t0
mtll0= round(np.mean(tll0),7)


tll=np.zeros(loops)
for l in range(loops):
    t0=time.time()
    ll(X, Z, sigmaX, sigmaA, K, D, N)
    t1=time.time()
    tll[l]=t1-t0
mtll= round(np.mean(tll),7)


times = np.array((mtll0,mtll))

columns = ['Time']
index = ['original ll function','Proposed ll function']

if not os.path.exists('latex_tables'):
    os.makedirs('latex_tables')
df = pd.DataFrame(times,columns=columns,index=index)
tab = df.to_latex()
text_file = open("latex_tables/llcomp.tex", "w")
text_file.write(tab)
text_file.close()
