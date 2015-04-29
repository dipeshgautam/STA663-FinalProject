from functions import ll0, ll
from sampler import sampler
from sampler0 import sampler0
from sampler_cy import sampler_cy
import numpy as np
import pandas as pd
import time
import os

X=np.load('Data/SimulatedData.npy')
N=X.shape[0]
D=X.shape[1]
sigmaX=1.
sigmaA=1.
alpha=1.
maxNew=4
niter=30

t0= time.time()
sampler0(X, niter, 0, sigmaX, sigmaA, alpha, N, D, maxNew)
t1=time.time()
elap1 = t1-t0

t0= time.time()
sampler(X, niter, 0, sigmaX, sigmaA,alpha, N, D, maxNew)
t1=time.time()
elap2 = t1-t0

t0= time.time()
sampler(X, niter, 0, sigmaX, sigmaA,alpha, N, D, maxNew)
t1=time.time()
elap3 = t1-t0


columns = ['Total Time']
index = ['Initial Code','Improved ll','Cythonized']

if not os.path.exists('latex_tables'):
    os.makedirs('latex_tables')

df = pd.DataFrame(np.hstack((elap1,elap2,elap3)),columns=columns,index=index)
tab = df.to_latex()
text_file = open("latex_tables/Runtimes.tex", "w")
text_file.write(tab)
text_file.close()