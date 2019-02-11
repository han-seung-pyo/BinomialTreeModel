# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:09:58 2019

@author: 한승표
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve, newton
import time ## 시간 측정하는 library. time.time(측정할 것)

def option_tree(s,k,r,T,sigma,N,option_type1,option_type2,tree_type,ann_div=None):
    dt = T/N
    v = np.zeros(N+1)
    vv =np.zeros(N+1)
    x =np.zeros(N+1)
    def Leisen_Reimer(s,k,sigma,r,ann_div,T,N):
        dt = T/N
        d1 = (np.log(s/k)+(r-ann_div+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(s/k)+(r-ann_div-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        
        def h_func(d):
            
            if d < 0:
               
               h = 0.5 - np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            else:
                
               h = 0.5 + np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6)))) 
               
            return h
        
        q_ = h_func(d1)
        q = h_func(d2)
        u = q_*np.exp((r-ann_div)*dt)/q
        d = (np.exp((r-ann_div)*dt)-q*u)/(1-q)
        
        return q,u,d
    
    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(dt))
        d = 1/u
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(dt))
        d = np.exp(r*T/N-sigma*np.sqrt(dt))
        
    elif tree_type == 'Rendleman':
        u = np.exp((r-ann_div-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-ann_div-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
        
    elif tree_type == 'LR':
        u = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[2]
        
        
    p = (np.exp(r*dt)-d)/(u-d)
    
    if option_type1=='call':    
        sign = 1
    else:
        sign = -1
    
    if option_type2 == 'European':
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = max(sign*(s1-k),0)
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            for i in range(N+1):
                v[i] = vv[i]
                
            
    elif option_type2=='American':
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = max(sign*(s1-k),0)
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                x[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
                s1 = s * (u**i) * (d**(N-j-i))
                vv[i] = max(x[i],sign*(s1-k))
            for i in range(N+1):
                v[i] = vv[i]
            
    return np.round(v[0],5)
#%%
def BD_method(s,k,r,sigma,div,T,N):
    dt = T/N
    u  = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-div)*dt)-d)/(u-d)
    v=np.zeros(N+1)
    vv = np.zeros(N+1)
    
    for i in range(N):
        s1 = s*(u**i)*(d**(N-1-i))
        d1_ = (np.log(s1/k)+((r-div+0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
        d2_ = (np.log(s1/k)+((r-div-0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
        v[i] = max(-s1 * np.exp(-div*dt)*norm.cdf(-d1_) + k*np.exp(-r*dt)*norm.cdf(-d2_),k-s1)
    
    for j in range(2,N+1):
        for i in range(N+1-j):
            vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            s1 = s * (u**i) * (d**(N-j-i))
            vv[i] = max(vv[i],(k-s1))
        for i in range(N+1):
            v[i] = vv[i]
            
    return np.round(v[0],6)