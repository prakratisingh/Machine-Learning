# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:05:00 2022
LOGISTIC REGRESSION
SINGLE AND MULTICLASS
@author: 91766
"""
#%%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
#making an array of (100,1) containing 1's
classes = 2
x, t = make_classification(100, 5, n_classes = classes, random_state= 40, n_informative = 2, n_clusters_per_class = 1)
x0 = np.ones((100,1))
#adding the x0 column to the x array
#%%
x= np.concatenate((x0,x),axis = 1)
x_train,x_test, t_train,t_test = train_test_split(x,t)
print(x.shape)
print(t.shape)
#%%
model = SGDClassifier(loss='log')
model = LogisticRegression(penalty='l2',fit_intercept=(True),solver='lbfgs',max_iter=100,multi_class='Sauto')
model.fit(x_train,t_train)
y = model.predict(x_test)
print(model.coef_, model.intercept_)
print(x.shape)
print(t)
#%%
W = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)),x_train.transpose()),t_train)
print(W.shape)
print(W)
#%%
predt = np.dot(x_test,W)
print(predt)
print(accuracy_score(t_test,predt))
#%%
'''
MULTICLASS CLASSIFICATION
'''
#%%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
#making an array of (100,1) containing 1's
classes = 3
x, t = make_classification(100, 5, n_classes = classes, random_state= 40, n_informative = 2, n_clusters_per_class = 1)
x0 = np.ones((100,1))
#adding the x0 column to the x array
#%%
x= np.concatenate((x0,x),axis = 1)
x_train,x_test, t_train,t_test = train_test_split(x,t)
print(x.shape)
print(t.shape)
#%%
model = LogisticRegression(penalty='l2',fit_intercept=(True),solver='newton-cg',max_iter=100,multi_class='multimonial')
model.fit(x_train,t_train)
y = model.predict(x_test)
print(model.coef_, model.intercept_)
print(x.shape)
print(t)
#%%
W = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train)),x_train.transpose()),t_train)
print(W.shape)
print(W)
#%%
predt = np.dot(x_test,W)
print(predt)
print(accuracy_score(t_test,predt))
#%%