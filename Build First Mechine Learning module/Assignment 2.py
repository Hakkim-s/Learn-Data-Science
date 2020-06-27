#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[11]:


titanic_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/First_ML_Model/master/titanic.csv", index_col = 'PassengerId')
titanic_data


# In[14]:


titanic_data.Sex.value_counts()


# In[21]:


titanic_data.Fare.isnull().sample()


# In[22]:


titanic_data.SibSp.value_counts()


# In[25]:


titanic_data.describe()


# In[89]:


#pro=su.pass/total.pass
342/891


# In[ ]:





# In[26]:


titanic_data.Fare.median()


# In[29]:


titanic_data.Survived.value_counts()


# In[32]:


titanic_data.Age.mode()


# In[83]:


survived_passengers = titanic_data[titanic_data.Survived == 1]
survived_passengers


# In[56]:


titanic_data.Embarked.value_counts()


# In[48]:


#to find Five highest fares of the passengers:
titanic_data.Fare.nlargest(10)


# In[49]:


titanic_data.Age.median()


# In[52]:


titanic_data.Name.unique()


# In[53]:


titanic_data.info()


# In[58]:


titanic_data.Name.value_counts().sum()


# In[71]:


print(100 * np.mean(titanic_data['Survived'] [titanic_data['Sex'] == 'male']))
print(100 * np.mean(titanic_data['Survived'] [titanic_data['Sex'] == 'female']))


# In[84]:


survived_passengers.Pclass.value_counts()


# In[85]:


print(100 * np.mean(titanic_data['Survived'] [titanic_data['Age'] < 18]))


# In[87]:


print(100 * np.mean(titanic_data['Survived'] [titanic_data['Age'] > 18]))


# In[88]:


survived_passengers.Embarked.value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




