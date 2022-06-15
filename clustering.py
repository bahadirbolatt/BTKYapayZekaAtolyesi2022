#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


data = pd.read_excel("aractablosu.xlsx")
data=data.dropna(axis='columns')


# In[10]:


data.head()


# In[11]:


plt.plot(data['Ortalama Sürüş Hızı (km/s)'],data['Ortalama Takip Mesafesi (m)'],"x")


# In[12]:


from sklearn.cluster import KMeans


# In[13]:


dt=pd.DataFrame()
dt['a']=data['Ortalama Sürüş Hızı (km/s)']
dt['b']=data['Ortalama Takip Mesafesi (m)']


# In[14]:


km = KMeans(
    n_clusters=6, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(dt)


# In[15]:


X=data['Ortalama Takip Mesafesi (m)']
Y=data['Ortalama Sürüş Hızı (km/s)']


# In[16]:


plt.scatter(Y,X, c=km.labels_, cmap='rainbow')

