# -*- coding: utf-8 -*-
"""
Created on Fri Feb 04 11:08:55 2022
@author: Prakrati Singh
"""
#%%
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
noise = [0,10,30,100]
for i in noise:
    #for making y a vector 
    x,t = make_regression(100,5,n_targets=3,shuffle=True,bias=0.0,noise=0,random_state=10)
    print(x.shape)
    print(t.shape)
#%%
    a=[]
#making an array of (100,1) containing 1's
x0 = np.ones((100,1))
#adding the x0 column to the x array
x= np.concatenate((x0,x),axis = 1)
print(x.shape)
#adding sigmoid function
x = 1/(1 + np.exp(-x))
#weights calculation
W = (x^t)^-1 * x^t* t
W = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),t)
print(W.shape)
print(W)
#using sklearn library
#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x,t)
y= reg.predict(x)
print("Score: ")
score = r2_score(t,y)
a.append(score)
print(a)
print(score)
import matplotlib.pyplot as plt
plt.xlabel('noise')
plt.ylabel('score')
plt.plot(score,noise)
#print(reg.coef_.transpose())
#print("regression intercept",reg.intercept_)

