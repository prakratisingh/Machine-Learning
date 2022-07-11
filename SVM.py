# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:38:34 2022

@author: 91766
"""

#%%
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

#%%
x, t = make_classification(n_classes = 2, random_state= 40, n_informative = 2, n_clusters_per_class = 1)

#%%
x = x[:,0:2]
print(x.shape)
x_train,x_test,t_train,t_test = train_test_split(x,t,random_state=60)

#%%
model = SVC(decision_function_shape="ovo")
model.fit(x_train,t_train)
#%%
y_pred = model.predict(x_test)

#%%
print(accuracy_score(t_test, y_pred))

#%%
plot_decision_regions(x_test, t_test, clf=model)
plt.xlabel("X test")
plt.ylabel("T test")
plt.title("Test")

#%%

plot_decision_regions(x_test, y_pred, clf=model)
plt.xlabel("X test")
plt.ylabel("Y pred")
plt.title("Predict")

#%%