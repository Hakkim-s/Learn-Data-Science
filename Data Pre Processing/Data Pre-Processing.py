#!/usr/bin/env python
# coding: utf-8

# ##### Outline:
#   Preprocessing/cleaning the data

# In[135]:


import numpy as np
import pandas as pd

train_data = pd.read_csv(r'D:\hakk study\Data Science\Data Pre processing\train - train_titanic.csv')
train_data1 = pd.read_csv(r'D:\hakk study\Data Science\Data Pre processing\train - train_titanic.csv')


# In[105]:


train_data.info()


# In[106]:


train_data1.info()


# In[95]:


train_data.head(3)


# In[140]:


#Dropping columns which are not useful
cols =['Name','Ticket','Cabin']
train_data = train_data.drop(cols, axis=1)
train_data1 = train_data1.drop(cols, axis=1)


# In[30]:


train_data.info()


# In[77]:


#Dropping Rows having missing values
train_data = train_data.dropna()
train_data.info()


# In[148]:


train_data1


# In[136]:


dummies = []

cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(train_data1[col]))
titanic_dummies = pd.concat(dummies, axis=1)    


# In[137]:


train_data1 = pd.concat((train_data1,titanic_dummies), axis=1)


# In[138]:


cols = ['Pclass', 'Sex', 'Embarked']
train_data1 = train_data1.drop(cols, axis=1)


# In[142]:


train_data1.info()


# In[145]:


#Taking care of issing values
train_data1['Age'] = train_data1['Age'].interpolate()


# In[147]:


train_data1.info()


# In[157]:


X = train_data1.values
y = train_data1['Survived'].values


# In[160]:


X = np.delete(X, 1, axis=1)


# In[159]:


X


# In[164]:


#Dividing data set into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state= 0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




