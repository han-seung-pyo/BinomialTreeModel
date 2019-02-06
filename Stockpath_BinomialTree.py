# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 07:27:53 2019

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
#%%
def BinomialTree(s,r,sigma,T,N, ann_div=None, divperiod=None, disp=None):
    df = pd.DataFrame(np.zeros((N+1,N+1)))
    df.loc[0,0] = s
    u  = np.exp(r*T/N+sigma*np.sqrt(T/N))
    d = np.exp(r*T/N-sigma*np.sqrt(T/N))

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
        
    if divperiod == 'M':
       div = ann_div/12 
       div_m = np.arange(1,13)
       div_time = np.floor(div_m*N/12)+1
       
       for j in range(1,N+1):
           for i in range(j+1):
               if i==0:
                   df.loc[i,j]=df.loc[i,j-1]*d
                   if j in div_time:
                      df.loc[i,j]=df.loc[i,j-1]*d*(1-div)
               else:
                   df.loc[i,j] = df.loc[i-1,j-1]*u
                   if j in div_time:
                      df.loc[i,j] = df.loc[i-1,j-1]*u*(1-div)

       
    elif divperiod == 'Q':
       div = ann_div/4
       div_q = np.arange(1,13,3)
       div_time = np.floor(div_q*N/12)+1
        
       for j in range(1,N+1):
           for i in range(j+1):
               if i==0:
                   df.loc[i,j]=df.loc[i,j-1]*d
                   if j in div_time:
                      df.loc[i,j]=df.loc[i,j-1]*d*(1-div)
               else:
                   df.loc[i,j] = df.loc[i-1,j-1]*u
                   if j in div_time:
                      df.loc[i,j] = df.loc[i-1,j-1]*u*(1-div)
                      
    elif divperiod== None:              
          for j in range(1,N+1):
              for i in range(j+1):
                  if i==0:
                       df.loc[i,j]=df.loc[i,j-1]*d
                  else:
                       df.loc[i,j] = df.loc[i-1,j-1]*u
    
    if disp :
       showTree(df) 

                                   
    return df

stock_path = BinomialTree(s,r,sigma,T,N,q,'Q',disp=None)
#%%
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
#%%
#Covvertable Bond Priing using Binomial Tree
#added Parameter
FV = 1000
Conversion_level = k/s
ratio = FV/k
coupon_rate = 0.0905
number_cpn = 12
coupon = T/number_cpn * coupon_rate *FV


        
#%%
#배당이 없을 경우
def Binomial_stock_tree(s,T,sigma,r,N):
    dt = T/N
    u  = round(np.exp(r*dt+sigma*np.sqrt(dt)),6)
    d = round(np.exp(r*dt-sigma*np.sqrt(dt)),6)
    df = pd.DataFrame(np.zeros((N+1,N+1)))
    df.loc[0,0] = s
    for j in range(1,N+1):
        for i in range(j+1):
            if i==0:
                df.loc[i,j]=df.loc[i,j-1]*d
            else:
                df.loc[i,j] = df.loc[i-1,j-1]*u
    return df

#stock_path = Binomial_stock_tree(s,T,sigma,r,N)

#배당이 있을 경우
def Binomial_stock_path_q(s,T,sigma,r,q,q_t,N):
    dt = T/N
    u  = round(np.exp(r*dt+sigma*np.sqrt(dt)),6)
    d = round(np.exp(r*dt-sigma*np.sqrt(dt)),6)
    df = pd.DataFrame(np.zeros((N+1,N+1)))
    div_t = np.floor(q_t*N/12)+1
    df.loc[0,0] = s
    for j in range(1,N+1):
        for i in range(j+1):
            if i==0:
                df.loc[i,j]=df.loc[i,j-1]*d
                if j in div_t:
                    df.loc[i,j]=df.loc[i,j-1]*d*(1-q)
            else:
                df.loc[i,j] = df.loc[i-1,j-1]*u
                if j in div_t:
                    df.loc[i,j] = df.loc[i-1,j-1]*u*(1-q)
    return df

#stock_path_considering_q = Binomial_stock_path_q(s,T,sigma,r,q,q_t,N)