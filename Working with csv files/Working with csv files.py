#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[8]:


import pandas as pd


# In[22]:


pd.read_csv("D:\hakk study\Data Science\Working with csv files\Effects-of-COVID19-on-trade-1-February-27-May-2020-Provisional.csv")


# In[47]:


a = pd.read_csv("D:\hakk study\Data Science\Working with csv files\Effects-of-COVID19-on-trade-1-February-27-May-2020-Provisional.csv")
a


# In[48]:


a.to_csv("D:\hakk study\Data Science\Working with csv files\Effects-of-COVID19-on-trade-1-February-27-May-2020-Provisional2.csv", index=False)


# In[62]:


greenhouse_data = pd.read_csv(r"D:\hakk study\Data Science\Working with csv files\UNdata_Export_20200604_085129204.csv")
greenhouse_data


# In[72]:


greenhouse_data = pd.read_csv(r"D:\hakk study\Data Science\Working with csv files\UNdata_Export_20200604_085129204.csv",index_col='Country or Area')
greenhouse_data


# In[37]:


greenhouse_data.head(10)


# In[39]:


greenhouse_data.tail(10)


# In[40]:


greenhouse_data.isnull()


# In[52]:


greenhouse_data.to_csv(r"D:\hakk study\Data Science\Working with csv files\UNdata_Export_20200604_085129204Two.csv")


# In[54]:


greenhouse_data['Year']


# In[75]:


greenhouse_data.Year.head(20)


# In[61]:


greenhouse_data.describe()


# In[63]:


greenhouse_data.max()


# In[64]:


greenhouse_data.min()


# In[66]:


greenhouse_data.mean()


# In[73]:


greenhouse_data.loc['Australia']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




