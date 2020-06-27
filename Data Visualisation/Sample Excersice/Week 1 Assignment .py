#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[138]:


import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt


# In[26]:


train_data = pd.read_csv(r'D:\hakk study\Data Science\Data Visualisation\Sample Excersice\Standard Metropolitan Areas Data - train_data - data.csv')
train_data1 = pd.read_csv(r'D:\hakk study\Data Science\Data Visualisation\Sample Excersice\Standard Metropolitan Areas Data - train_data - data.csv')


# In[27]:


train_data


# In[28]:


train_data.describe()


# In[29]:


train_data.info()


# In[45]:


[train_data['income'].min()]


# In[46]:


[train_data['land_area'].mean()]


# In[47]:


[train_data['hospital_beds'].isnull()]


# In[48]:


[train_data['region'].mode()]


# In[106]:


region_len4 = train_data[train_data['region'] >= 4 ]
region_len4


# In[85]:


region_len4.count()


# In[61]:


[train_data['crime_rate'].mean()]


# In[62]:


train_data.corr()


# In[86]:


region_len4.head(5)


# In[101]:


region_len4['crime_rate'] == 85.62


# In[99]:


region_len4


# In[110]:


region_len3 = train_data[train_data['region'] == 3 ]
region_len3


# In[133]:


region_len1 = train_data[train_data['region'] == 1 ]
region_len1


# In[130]:


region_len1[region_len1['crime_rate'] >= 54.16]
# region_len1 = train_data[(train_data['region'] == 1)  & (train_data['crime_rate'] >= 54.16)]


# In[165]:


x = train_data
plt.plot(x.land_area,x.crime_rate)
plt.xlabel('land_area')
plt.ylabel('crime_rate')
plt.show()


# In[166]:


plt.scatter(x.physicians,x.hospital_beds)


# In[168]:


x = train_data
# plt.bar(x.region,color)
# plt.show()
train_data['region'].value_counts().plot(kind='bar');


# In[164]:


x = train_data
plt.hist(x.income)
plt.show()


# In[ ]:





# In[ ]:




