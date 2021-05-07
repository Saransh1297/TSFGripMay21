#!/usr/bin/env python
# coding: utf-8

# ## Author-Pande Saransh
# ## The Sparks Foundation
# ## Data Science and Business Analytics
# ## Task 2-Unsupervised ML- Kmeans Cluster Analysis
# ### Problem Statement- From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually. 
# ### dataset- https://bit.ly/3kXTdox

# ### Importing standard ML libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as datasets


# ### Understanding the data

# In[4]:


iris_filepath=("C://Users/Poorva/Desktop/saransh/Iris.csv")


# In[20]:


iris_df=pd.read_csv(iris_filepath)
iris_df.head()


# In[21]:


iris_df.shape


# In[29]:


iris_df.dtypes


# In[23]:


iris_df.isnull().sum()


# In[30]:


iris_df.describe()


# ### Plotting for understanding the distribution

# In[25]:


figsise=(20,20)
sns.pairplot(iris_df,hue="Species")
plt.show()


# ### Finding number of clusters (K) by Elbow Method

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans 


# In[65]:


#Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
   kmeans = KMeans(n_clusters=i, init ='k-means++', 
                    random_state = 0)
   kmeans.fit(x)
   wcss.append(kmeans.inertia_)
   
plt.plot(range(1, 11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('TWCSS') # Total Within cluster sum of squares
plt.show()


# ### We choose no of clusters as '3'

# ### Creating and Plotting the kmeans clusters distribution visuals

# In[66]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
               random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[67]:


y_kmeans


# In[68]:


kmeans.cluster_centers_


# In[87]:


plt.figure(figsize=(5,5))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 40, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 40, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 40, c = 'green', label = 'Iris-virgnica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 75, c = 'black', label = 'Centroids')

plt.xlabel('Sepal length cm')
plt.ylabel('Petal length cm')
plt.title('k-means Clustering')
plt.legend()


# ### Thankyou! 
