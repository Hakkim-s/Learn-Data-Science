#!/usr/bin/env python
# coding: utf-8

# In[8]:


from statistics import mode
mode([5, 17, 23, 31, 43, 49, 57, 17, 57, 17])


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


train_data=pd.read_csv("Standard Metropolitan Areas Data - train_data - data.csv")


# In[62]:


train_data1=pd.read_csv("Standard Metropolitan Areas Data - train_data - data.csv")


# In[63]:


train_data1


# In[64]:


train_data= train_data1


# In[65]:


train_data


# In[ ]:





# In[35]:


count=0
if train_data.region ==4:
    count+=1
else:
    pass


# In[61]:


train_data


# In[ ]:





# In[13]:


train_data.describe()


# In[34]:


mode(train_data.region)


# In[ ]:





# In[ ]:





# In[14]:


train_data.info()


# In[16]:


train_data.head()


# In[17]:


train_data.tail()


# In[22]:


train_data['hospital_beds'].isnull().sum()


# In[30]:


plt.scatter(train_data.crime_rate,train_data.region)


# In[32]:


x=train_data
plt.plot(x.land_area, x.crime_rate) 
plt.show()


# In[68]:


train_data[(train_data['region'] ) & (train_data['land_area'] >= 5000)]


# In[ ]:


train_data['crime_rate'] >)


# In[ ]:





# In[73]:


train_data[(train_data['region'] >3)]


# In[79]:


qq=train_data[(train_data['region'] <=1)]


# In[80]:


qq


# In[81]:


qq[qq['crime_rate']>=54.16]


# In[84]:


plt.scatter(train_data.physicians,train_data.hospital_beds)
plt.title('Plot of hospital_beds,physicians') # Adding a title to the plot
plt.ylabel("hospital_beds") # Adding the label for the horizontal axis
plt.xlabel("physicians") # Adding the label for the vertical axis
plt.show()


# In[87]:


train_data['region'].value_counts().plot(kind='bar');


# In[89]:


plt.hist(train_data.income)
plt.show()


# In[91]:


train_data.corr()


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




