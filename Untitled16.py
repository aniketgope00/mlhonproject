#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
dataframe = pd.read_csv('dataset_small.csv')
data = dataframe.values
X,y = data[:,:-1],data[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[10]:


#Naive Bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_absolute_error
clf = BernoulliNB()
clf.fit(X_train,y_train)
yhat = clf.predict(X_test)
mae = mean_absolute_error(y_test,yhat)
print('MAE = %.3f' %mae)
print('Accuracy = %.2f'%(1-mae))


# In[13]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
model = LogisticRegression(random_state=None,max_iter=10000)
model.fit(X_train,y_train)
yhat = model.predict(X_test)
mae = mean_absolute_error(y_test,yhat)
print('MAE = %.3f' %mae)
print('Accuracy = %.2f'%(1-mae))


# In[15]:


#Random Forrest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
clf = RandomForestClassifier(max_depth = 3,random_state = 0)
clf.fit(X_train,y_train)
yhat = clf.predict(X_test)
mae = mean_absolute_error(y_test,yhat)
print('MAE = %.3f' %mae)
print('Accuracy = %.2f'%(1-mae))


# In[23]:


#Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
clf = MLPClassifier(random_state=1,max_iter=300)
clf.fit(X_train,y_train)
yhat = clf.predict(X_test)
mae = mean_absolute_error(y_test,yhat)
print('MAE = %.3f' %mae)
print('Accuracy = %.2f'%(1-mae))


# In[ ]:


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
regr = make_pipeline(StandardScaler(),SVR(c=1.0,epsilon=0.2))
regr.fit(X_train,y_train)
yhat = regr.predict(X_test)
mae = mean_absolute_error(y_test,yhat)
print('MAE = %.3f' %mae)
print('Accuracy = %.2f'%(1-mae))


# In[ ]:




