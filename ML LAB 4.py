# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:31:38 2022

@author: Prakrati Singh
"""
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
#storing the noise values in an array
noise = [0,10,30,100]
#storing r2 score of training data from Linear regression
SCORE_train = []
#storing r2 score of testing data from Linear regression
SCORE_test = []
#storing r2 score of training data from Ridge regression
SCORE_ridgetrain =[]
#storing r2 score of testing data from Ridge regression
SCORE_ridgetest=[]

for i in noise:
    x,t = make_regression(100,5,n_targets=1,shuffle=True,bias=0.0,noise=i,random_state=10)
    data = LinearRegression()
    #Split arrays or matrices into random train and test subsets.
    x_train,x_test,t_train,t_test = train_test_split(x,t)
    '''
    Fit linear model.
    Parameters
    X{array-like, sparse matrix} of shape (n_samples, n_features)
    Training data.
    yarray-like of shape (n_samples,) or (n_samples, n_targets)
    Target values.
    '''
    x_train,x_test,t_train,t_test = train_test_split(x,t)
    ridge_reg = Ridge(alpha=0.5).fit(x_train,t_train)
    #predicting with the help of Ridge Regression 
    y_pred_train = ridge_reg.predict(x_train)
    y_pred_test = ridge_reg.predict(x_test)
    
    score = r2_score(t_train,y_pred_train)
    SCORE_train.append(score)
    score = r2_score(t_test,y_pred_test)
    SCORE_test.append(score)
    
    #calculating the r2 score for training and testing data set
    score = r2_score(t_train,y_pred_train)
    SCORE_ridgetrain.append(score)
    score = r2_score(t_test,y_pred_test)
    SCORE_ridgetest.append(score)
    
import matplotlib.pyplot as plt
#r2 score with linear regression
'''
Linear Regression is a machine learning algorithm based on supervised learning. 
It performs a regression task.
'''
print("Score with linear regression : ")
print(SCORE_train)
print(SCORE_test)

'''
Ridge regression is a method of estimating the 
coefficients of multiple-regression models in scenarios 
where linearly independent variables are highly correlated
'''
#r2 score with ridge regression
print("Score with ridge regression : ")
print(SCORE_ridgetrain)
print(SCORE_ridgetest)


#plotting r2 score vs noise in Linear regression
plt.plot(noise,SCORE_train)
plt.plot(noise,SCORE_test)
plt.ylabel("R2 SCORE")
plt.xlabel("NOISE")
plt.title("Linear Regression")
plt.show()

#plotting r2 score vs noise in Ridge regression
plt.plot(noise,SCORE_ridgetrain)
plt.plot(noise,SCORE_ridgetest)
plt.ylabel("R2 SCORE")
plt.xlabel("NOISE")
plt.title("Ridge Regression")
plt.show()
