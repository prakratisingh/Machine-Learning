# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:23:57 2022

@author: 
"""
#%%
from sklearn.datasets import make_classification
from sklearn.datasets import KMeans
import matplotlib.pyplot as plt
#%%
x,t = make_classification(100,4,random_state=25)

print(x.shape)
#%%
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x)
    distortions.append(kmeanModel.inertia_)
#%%
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal clusters')
plt.show()
#%%
model = KMeans(n_clusters=4, random_state=80)
# Fit into our dataset fit
kmeans_predict = model.fit_predict(x)

print(kmeans_predict)
#%%
plt.scatter(x[kmeans_predict == 0, 0], x[kmeans_predict == 0, 1], s = 100, c = 'red')
plt.scatter(x[kmeans_predict == 1, 0], x[kmeans_predict == 1, 1], s = 100, c = 'blue')
plt.scatter(x[kmeans_predict == 2, 0], x[kmeans_predict == 2, 1], s = 100, c = 'green')
plt.scatter(x[kmeans_predict == 2, 0], x[kmeans_predict == 2, 1], s = 100, c = 'orange')
#%%