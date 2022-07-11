# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:43:54 2022

@author: Prakrati Singh
SAP - 500082638
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
X,t = make_regression(100,1,shuffle=True, bias=0.0,noise=0,random_state=2)
#MAximum and min values
print("Min X ",min(X))
print("Max X ",max(X))
print("Min t ",min(t))
print("Max t ",max(t))
a=X
#mean and varience
print("Mean of X ", np.mean(X))
Xm = np.mean(X)
print("Mean of t ", np.mean(t))
tm = np.mean(t)

print("Standard dev of X ",np.std(X))
print("Standard dev of t ",np.std(t))

#scatter plot
plt.scatter(X,t)
plt.show()

# B0 = y mean - m* x mean
#re shaping
print(np.shape(X))
X = X.flatten()
print(np.shape(X))
print(np.shape(t))

b1 = (sum((X-Xm)*(t-tm)))/(sum((X-Xm)**2))
print(b1)
b0 = tm - b1*Xm
print(b0)
#using sklearn library
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(a,t)
print("regression score: ",reg.score(a,t))

print("regression coefficient",reg.coef_)
print("regression intercept",reg.intercept_)