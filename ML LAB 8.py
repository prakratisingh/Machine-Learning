# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:21:11 2022
DECISION TREE

@author: 91766
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

#%%
classes = 3
X,t = make_classification(100,5,n_classes = classes,random_state = 40, n_informative = 2, n_clusters_per_class = 1)

X_train, X_test, t_train, t_test = train_test_split(X,t)

#%%

model = tree.DecisionTreeClassifier(max_depth =2)
model.fit(X_train,t_train)

tree.plot_tree(model)
#%%

y_pred=model.predict(X_test)
#calculating accuracy of DT
print(accuracy_score(t_test, y_pred))

#%%