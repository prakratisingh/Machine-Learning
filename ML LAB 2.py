# -*- coding: utf-8 -*-
"""
Created on Fri Feb 04 11:08:55 2022
@author: Prakrati Singh
"""
#%%
import numpy as np
from sklearn.datasets import make_regression
#making the dataset using make_regression() function
#make_regression produces regression targets as an optionally-sparse 
#random linear combination of random features, with noise
x,t = make_regression(100,5,n_targets=3,shuffle=True,bias=0.5,noise=0,random_state=2)
#%%
'''
Generate a random regression problem.
Description of parameters:
1. n_samples int, default=100
The number of samples.
2. n_features int, default=100
The number of features.
3. n_targets int, default=1
The number of regression targets, i.e., the dimension of the y output vector associated with a sample.
4. bias float, default=0.0
The bias term in the underlying linear model.
5.noise float, default=0.0
The standard deviation of the gaussian noise applied to the output.
6.shuffle bool, default=True
Shuffle the samples and the features.
7.random_state int, RandomState instance or None, default=None
Determines random number generation for dataset creation
'''
#%%
print(x.shape)
print(t.shape)
#%%
#making an array of (100,1) containing 1's
x0 = np.ones((100,1))
#adding the x0 column to the x array
x= np.concatenate((x0,x),axis = 1)
print(x.shape)
#%%
#weights calculation
#W = (x^Tx)^-1 * x^T* t
W = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),t)
print(W.shape)
print(W)
#%%
#calculating the values of y 
y=0
for i in range(0,x.shape[1]):
    y = y + W[i]*x[i]
print(y.shape)
print(y)
#%%
#using sklearn library
'''
LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the 
residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

Parameters

fit_interceptbool, default=True
Whether to calculate the intercept for this model. 
If set to False, no intercept will be used in calculations (i.e. data is expected to be centered). 
'''
#%%
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x,t)
#coef_ attribute to view the coefficients.For every one-unit increase in [X variable], 
#the [y variable] increases by [coefficient] when all other variables are held constant.
print(reg.coef_.transpose())
#The intercept (often labeled as constant) is the point where the function crosses the y-axis. In some analysis, the regression model only becomes
#significant when we remove the intercept, and the regression line reduces to Y = bX + error.
print("regression intercept",reg.intercept_)

#%%