# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:37:55 2022

@author: melih
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_excel("TümörVerisi.xlsx")

info = data.info()
describe = data.describe()
corr = data.corr()

data.dropna(axis=0,inplace=True)
data.info()


sn.pairplot(data)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(pd.concat((data.iloc[:,1],data.iloc[:,4]),axis=1))

veri = np.concatenate((X_train,data.iloc[:,-1].values.reshape(-1,1)),axis=1)
a = (veri[:,-1] == 0).nonzero()
b = (veri[:,-1] == 1).nonzero()
plt.figure()
plt.scatter(veri[a,0],veri[a,1])
plt.scatter(veri[b,0],veri[b,1])

from sklearn.decomposition import PCA
pca = PCA(n_components= 2 , whiten=True) #2 temel bileşen, whiten = normalize
pca.fit(veri)

x_pca = pca.transform(veri)
y = veri[:,-1]

plt.figure()
plt.scatter(x_pca[a,0],x_pca[a,1])
plt.scatter(x_pca[b,0],x_pca[b,1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3,metric="minkowski")
knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

knn_conf = confusion_matrix(y_test,y_pred_knn)
knn_acc = accuracy_score(y_test, y_pred_knn)
print(knn_acc)