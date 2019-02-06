# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 05:43:54 2019

@author: 한승표
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

#Parmeter
s = 102.8
k=90.46
sigma = 0.2704
r = 0.0271
q = 0.0184
T= 1
N = 50
q_t= np.array([1,4,7,10])
#Option using Binomial Tree
def Option_BinomialTree(stock_path,k,r,T,sigma,N,option_type1,option_type2,disp=None):
    dt = T/N
    u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
    d = np.exp(r*T/N-sigma*np.sqrt(T/N))
    p = (np.exp(r*dt)-d)/(u-d)
    df = pd.DataFrame(np.zeros((N+1,N+1)))
    
    def showTree(tree):
        t = np.linspace(T/N, T, N+1)
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        for i in range(len(t)):
            for j in range(i+1):
                ax.plot(t[i], tree[i][j], '.b')
                if i<len(t)-1:
                    ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j]], '-b')
                    ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j+1]], '-b')
        fig.show()
    
    if option_type1=='C':    
        sign = 1
    else:
        sign = -1
    
    if option_type2 == 'European':
        for j in np.arange(N,-1,-1):
            for i in range(j+1):
                if j==N:
                    df.loc[i,j]= max(sign*(stock_path.loc[i,j]-k),0)
                else:
                    df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1])
    
    else:
        for j in np.arange(N,-1,-1):
            for i in range(j+1):
                if j==N:
                    df.loc[i,j]= max(sign*(stock_path.loc[i,j]-k),0)
                else:
                    df.loc[i,j] = max(np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1]),sign*(stock_path.loc[i,j]-k))
        
        
    if disp :
        showTree(df)
        
    return df

European_C = Option_BinomialTree(stock_path,k,r,T,sigma,N,'C','European')
American_C = Option_BinomialTree(stock_path,k,r,T,sigma,N,'C','American')