# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 07:20:49 2019

@author: 한승표
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

def DAO_Binomial(s,k,B,r,q,sigma,T,N,option_type,tree_type,barrier_type=None,Barrier_number=None,Barrier_time=None):
    dt = T/N
    v = np.zeros(N+1)
    vv =np.zeros(N+1)
    cond=0
    
    def Leisen_Reimer(s,k,sigma,r,q,T,N):
        dt = T/N
        d1 = (np.log(s/k)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(s/k)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        
        def h_func(d):
            
            if d < 0:
               
               h = 0.5 - np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            else:
                
               h = 0.5 + np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6)))) 
               
            return h
        
        q_ = h_func(d1)
        q = h_func(d2)
        u = q_*np.exp((r-q)*dt)/q
        d = (np.exp((r-q)*dt)-q*u)/(1-q)
        
        return q,u,d
    
    
    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(dt))
        d = 1/u
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(dt))
        d = np.exp(r*T/N-sigma*np.sqrt(dt))
        
    elif tree_type == 'Rendleman':
        u = np.exp((r-q-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-q-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
        
    elif tree_type == 'LR':
        u = Leisen_Reimer(s,k,sigma,r,q,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,q,T,N)[2]
        
        
    p = (np.exp(r*dt)-d)/(u-d)
    

    if option_type=='call':    
        sign = 1
    else:
        sign = -1
        
    for i in range(N+1):
        s1 = s* (u**i)*(d**(N-i))
        v[i] = max(sign*(s1-k),0)
        
        if s1 >B and cond == 0:
            sk = s1
            sk_under = s *u**(i-1)*d**(N-(i-1))
            lambda_cal = (sk-B)/(sk-sk_under)
            cond=1
            
    for j in range(1,N+1):
        for i in range(N+1-j):
            vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            s1 = s * (u**i) * (d**(N-j-i))
            if s1<B:
                vv[i]=0
                
        for i in range(N+1):
            v[i] = vv[i]
            
    if barrier_type=='Discrete':
        dn= N/Barrier_number
        barrier_node = dn * Barrier_time #정수 일수도, 아닐 수도 있다!
        barrier_node_ad = barrier_node.astype(int)
        barrier_index = -1
        
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = max(sign*(s1-k),0)
        
            if s1 >B and cond == 0:
                sk = s1
                sk_under = s *u**(i-1)*d**(N-(i-1))
                lambda_cal = (sk-B)/(sk-sk_under)
                cond=1
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
                s1 = s * (u**i) * (d**(N-j-i))
                
                if j in barrier_node_ad:
                    if s1*np.exp(r*(barrier_node[barrier_index]-j)*dt)<B:
                        vv[i] = 0
            
            if j in barrier_node_ad:
                barrier_index -=1
                
            for i in range(N+1):
                v[i] = vv[i]
        
    return v[0], lambda_cal
