
# coding: utf-8

# In[ ]:


# importing important libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[ ]:


# reading data

data = pd.read_sas('DR1IFF_I.XPT')


# In[ ]:


data = data.dropna()


# In[ ]:


# finding correlation of all features with CHO

corelation = data[data.columns[1:]].corr()['DR1ICARB'][:]


# In[ ]:


# finding highly correlated values with CHO

print(corelation[abs(corelation.values) > 0.5])


# In[ ]:


# creating new df with inportant features

data1 = data[['DR1IKCAL', 'DR1ICARB', 'DR1ISUGR', 'DR1IFIBE', 'DR1_020', 'DR1CCMTX', 'DR1_030Z']].copy()


# In[ ]:


data1['DR1_020'] = data1['DR1_020']/3600


# In[ ]:


# creating correlation matrix

cor = data1.corr()


# In[ ]:


plt.rcParams.update({'figure.figsize': (10,10)})
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cor, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(cor.index)
ax.set_yticklabels(cor.index)
#ax.figure()
plt.show()


# In[ ]:


# creating training data

X = data1[['DR1_020', 'DR1IKCAL', 'DR1ISUGR', 'DR1IFIBE', 'DR1_030Z', 'DR1CCMTX']].copy()
y = data1[['DR1ICARB']].copy()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[ ]:


lr = LinearRegression().fit(X_train,y_train)
print(lr.score(X_train, y_train)*100)
print(lr.score(X_test, y_test)*100)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred)*100)
print(mean_squared_error(y_test, y_pred))


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


plt.plot(y_test.values[0:100], color="black", label="actual")
plt.plot(y_pred[0:100], color="red", label = "predicted")
plt.legend()


# In[ ]:


plt.plot(y_test.values[0:100], label="actual")
plt.plot(y_pred[0:100], label = "predicted")
plt.legend()
plt.ylabel('CHO level')


# In[ ]:


plt.plot(y_test.values[0:100], color="black", label="actual")
plt.scatter(range(0,100), y_pred[0:100], color="red", label = "predicted")
plt.legend()


# In[ ]:


lr1 = LinearRegression().fit(X,y)
print(lr1.score(X, y)*100)
y_pred = lr1.predict(X)
print(r2_score(y, y_pred)*100)
print(mean_squared_error(y, y_pred))


# In[ ]:


from sklearn.manifold import TSNE
fashion_tsne = TSNE(random_state=0).fit_transform(data1)


# In[ ]:


fashion_tsne
plt.scatter(fashion_tsne[:,0], fashion_tsne[:,1])


# In[ ]:


fashion_tsne

