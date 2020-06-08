#!/usr/bin/env python
# coding: utf-8

# 

# In[3]:


get_ipython().system('pip install missingno')


# In[6]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble,tree , linear_model
import missingno  as msno 


# In[11]:


train = pd.read_csv(r'D:\hakk study\Data Science\Exploratory Data Analysis\train1 - train_houseprices.csv')
test = pd.read_csv(r'D:\hakk study\Data Science\Exploratory Data Analysis\test - test_houseprices.csv')


# In[13]:


train


# In[16]:


train.describe()


# In[17]:


train.head()


# In[18]:


train.tail()


# In[29]:


train.shape, test.shape


# In[41]:


#Numeric features in the train dataset
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns


# In[42]:


numeric_features.head()


# In[56]:


#list of variables that contains year information
year_feature = [feature for  feature in numeric_features if 'Yr' in feature or 'Year' in feature]
year_feature


# In[57]:


# Let us explore the contents of temporal  variables
for feature in year_feature:
    print(feature, train[feature].unique())


# In[66]:


for feature in year_feature:
    if feature!='YrSold':
        data=train.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


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




