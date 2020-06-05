#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
print(pandas.__version__)


# In[6]:


import numpy as np
import pandas as pd


# In[45]:


#Series
data = pd.Series([0.25,0.5,0.75,1.0])
data


# In[10]:


data.index


# In[16]:


data.values


# In[28]:


data[1]


# In[27]:


data[1:3]


# In[32]:


data = pd.Series([0.25,0.5,0.75,1.0], index=['a','b','c','d'] )
data


# In[34]:


data['c']


# In[85]:


data = pd.Series([3,2,'5',7.0,"king"], index=[3,4,4,4,0] )
data


# In[86]:


#Dictionary
states_populationUSA_dict={
    'Texas':12121212122,
    "New York":32323989877,
    'Florida':888732654
}
population=pd.Series(states_populationUSA_dict)
population


# In[42]:


population['New York']


# In[46]:


population['Texas':'New York']


# In[59]:


#Data Frame
states_populationUSA_dict={
    'Texas':43434,
    "New York":75454,
    'Florida':21233
}
area = pd.Series(states_populationUSA_dict)
area


# In[61]:


states = pd.DataFrame({'population':population,
                      'area':area})
states


# In[69]:


states.index


# In[68]:


states.columns


# In[87]:


#Index
ind = pd.Index([3,4,2,1,4,5,3,6])
ind


# In[74]:


ind[6]


# In[80]:


ind[::2]


# In[83]:


ind.size,ind.shape,ind.ndim,ind.dtype


# In[89]:


#Data indexing and selection
#Data selection in series

data = pd.Series([0.25,0.5,0.75,1.0], index=['a','b','c','d'])
data


# In[90]:


data['e'] = 1.25
data


# In[94]:


#slicing by explicit index
data['a':'c']


# In[92]:


#slicing by implicit integer index
data[0:2]


# In[93]:


data[['a','e']]


# In[99]:


#loc and iloc
data=pd.Series(['a','b','c'], index=[1,2,3])
data


# In[103]:


# loc explicit index
data.loc[1]


# In[101]:


data.loc[1:3]


# In[104]:


data.iloc[1]


# In[105]:


data.iloc[1:3]


# In[117]:


arae = pd.Series({
    'Texas':43434,
    "New York":75454,
    'Florida':21233
})
pop = pd.Series({
    'Texas':434211234,
    "New York":721245454,
    'Florida':212645633
})
data = pd.DataFrame({'area':area,"pop":pop})
data


# In[111]:


data.area
data['area']


# In[118]:


data['density']=data['pop'] / data['area']
data


# In[119]:


data.values


# In[121]:


data.iloc[:2,:3]


# In[130]:


data.loc[:'New York',:'pop']


# In[134]:


data[data.density >10000]


# In[140]:


#Handling missing values
data = pd.Series([1,np.nan,2,'hello',None])


# In[153]:


data.isnull()


# In[152]:


data[data.isnull()]


# In[154]:


data[data.notnull()]


# In[155]:


data.dropna()


# In[160]:


df =pd.DataFrame([[1,np.nan,2],
              [2,3,5],
              [np.nan,4,6]])
df


# In[163]:


df.dropna()


# In[168]:


df.dropna(axis=0)


# In[167]:


df.dropna(axis=1)


# In[170]:


df[3]=np.nan
df


# In[171]:


df.dropna(axis='columns',how='all')


# In[177]:


#Filling null values
data =pd.Series([1,np.nan,2,None,3], index=list('abcde'))
data


# In[186]:


data.fillna(9)


# In[187]:


#forward-fill
data.fillna(method='ffill')


# In[189]:


#back-fill
data.fillna(method='bfill')


# In[190]:


df


# In[196]:


df.fillna(method='ffill',axis=0)


# In[197]:


df.fillna(method='ffill',axis=1)


# In[208]:


#pandas string operations
data=['hakk','abul','jerina','SHAik']
[s.capitalize() for s in data]


# In[209]:


names = pd.Series(data)
names


# In[210]:


#concat and append
ser1 = pd.Series(['A','B',"C"],index=[1,2,3])
ser2 = pd.Series(['D','E',"F"],index=[4,5,6])
pd.concat([ser1,ser2])


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




