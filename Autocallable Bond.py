# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:43:39 2019

@author: 한승표
"""


import numpy as np
import pandas as pd
import matplot.pyplot as plt
#%%
#Coupon Payment 
def cpn_period(cpn_type=None):
                
    if cpn_type == 'M':
        T = np.arange(1,13,1)
        term = 12
        period = T/term
        
    elif cpn_type == 'Q':
        T = np.arange(1,13,3)
        term = 12
        period = T/term
        
    elif cpn_type == 'Y':
        T = 12
        term = 12
        period = T/term
                      
    return np.array(period)   

#%%
#Parameter
s = 102.8
k=90.46
sigma = 0.2704
r = 0.0271
q = 0.0184
T= 1
N = 50
q_t= np.array([1,4,7,10])
cpn_time = cpn_period('M')
FV = 1000
ratio = FV/k
coupon_rate = 0.0905
autocall_period = np.array([3,6,9])/12
stock_path = BinomialTree(s,r,sigma,T,N,q,'Q',disp=None)
#%%
def Autocallable_bond(stock_path,FV,ratio,coupon_rate,T,N,sigma,cpn_time,autocall_period,disp=None):
    
    dt = T/N
    u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
    d = np.exp(r*T/N-sigma*np.sqrt(T/N))
    p = (np.exp(r*dt)-d)/(u-d)
    df=pd.DataFrame(np.zeros((N+1,N+1)))
    cpn = coupon_rate * FV / np.size(cpn_time)
    
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
    
    for j in np.arange(N,-1,-1):
        if j==N:
            cpn_time = cpn_time[:-1]     
        for i in range(j+1):
            if j==N:
                if stock_path.loc[i,j] < k:
                     df.loc[i,j]=ratio*stock_path.loc[i,j] + cpn
                else:
                    df.loc[i,j]= FV + cpn
            else:
                if j*dt < cpn_time[-1]:
                    if j*dt < autocall_period[-1]:
                        if stock_path.loc[i,j]*np.exp(r*(autocall_period[-1]-j*dt))<stock_path.loc[0,0]:
                            df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1]) + cpn*np.exp(-r*(autocall_period[-1]-j*dt))
                        else:
                            df.loc[i,j]= (FV+cpn)*np.exp(-r*(autocall_period[-1]-j*dt))
                        
                    else:
                        df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1]) + cpn*np.exp(-r*(cpn_time[-1]-j*dt))
                else:
                     df.loc[i,j] = np.exp(-r*dt)*(p*df.loc[i+1,j+1]+(1-p)*df.loc[i,j+1])
                     
        if j*dt <cpn_time[-1]:
            cpn_time = cpn_time[:-1]
            
        if np.size(cpn_time)==0:
            
            cpn_time = np.array([0])
            
        if j*dt <autocall_period[-1]:
            autocall_period = autocall_period[:-1]
            
        if np.size(autocall_period)==0:
            autocall_period = np.array([0])

    if disp :
       showTree(df) 

    return df

x =  Autocallable_bond(stock_path,FV,ratio,coupon_rate,T,N,sigma,cpn_time,autocall_period)