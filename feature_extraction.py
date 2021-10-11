#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
df = pd.read_csv('dataset_small.csv')
df.head()


# In[38]:


df.isnull().sum()/len(df)*100


# In[19]:


df.cov()


# In[26]:


df.isin([0]).sum(axis=1)


# In[47]:


import pandas as pd
from sklearn.preprocessing import normalize
normalize = normalize(df)
df_scaled = pd.DataFrame(normalize)
variance = df.var()
columns = df.columns
variable = []
var_limit = 0.01*(sum(variance)/len(variance))
for i in range(len(variance)):
    if variance[i]>=var_limit:
        variable.append(columns[i])
new_df = df[variable]
new_df


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
cr = new_df.corr()
f,ax = plt.subplots(figsize=(12,10))
mask = np.triu(np.ones_like(cr,dtype=bool))
cmap = sns.diverging_palette(230,20,as_cmap = True)
sns.heatmap(cr,annot=True,mask=mask,cmap=cmap)


# In[ ]:


df.to_csv(index=False)


# In[ ]:




