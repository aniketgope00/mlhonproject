#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
dataframe = pd.read_csv('dataset_small.csv')
data = dataframe.values
X,y = data[:10000,:-1],data[:10000,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[13]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
regr = make_pipeline(StandardScaler(),LinearSVC(random_state=0,tol=1e-5,max_iter=10000,dual=True))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    regr.fit(X_train,y_train)
yhat = regr.predict(X_test)
mae = mean_absolute_error(y_test,yhat)
print("MAE: %.3f"%mae)


# In[ ]:




