# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:05:06 2022
FIRST CLASSIFICATION MODEL
@author: PRAKRATI SINGH (500082638)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
classes = 2
x, t = make_classification(100, 5, n_classes = classes, random_state= 40, n_informative = 2, n_clusters_per_class = 1)
'''
Generate a random n-class classification problem.
This initially creates clusters of points normally distributed (std=1) about vertices of an 
n_informative-dimensional hypercube with sides of length 2*class_sep and assigns an equal number of clusters to each class.
It introduces interdependence between these features and adds various types of further noise to the data.
 '''
res = np.zeros((t.shape[0], classes), dtype=int)
res[np.arange(t.shape[0]), t] = 1
print(res)
#making an array of (100,1) containing 1's
x0 = np.ones((100,1))
#adding the x0 column to the x array
x= np.concatenate((x0,x),axis = 1)
print(x.shape)
#Split arrays or matrices into random train and test subsets.
x_train,x_test, res_train,res_test = train_test_split(x,res)
print(x.shape)
print(res.shape)
print(res)
#adding sigmoid function
x= 1/(1 + np.exp(-x))
#weights calculation
#W = (x^Tx)^-1 * x^T* t
W = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)),x_train.transpose()),res_train)
print(W.shape)
print(W)
predt = np.dot(x_test,W)
print(predt)
#decision stage
for i in range(len(predt)):
    if(predt[i][0]>=predt[i][1] and predt[i][2]): 
        predt[i][0]=1
        predt[i][1]=0
        predt[i][2]=0
    elif(predt[i][1]>=predt[i][0] and predt[i][2]):
        predt[i][1]=1
        predt[i][0]=0
        predt[i][2]=0
    else:
        predt[i][2]=1
        predt[i][0]=0
        predt[i][1]=0
print(predt)
print(len(predt))
print(predt.shape)
#calculating r2_score
score=r2_score(res_test,predt)
print(score)
#calculating accuracy
print(accuracy_score(res_test,predt))
