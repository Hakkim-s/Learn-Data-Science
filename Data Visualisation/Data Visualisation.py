#!/usr/bin/env python
# coding: utf-8

# In[3]:





# In[124]:


pip install matplotlib


# In[120]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[10]:


x = pd.read_csv("D:\hakk study\Data Science\Data Visualisation\Standard Metropolitan Areas Data - train_data - data.csv")
x


# In[11]:


x.head()


# In[17]:


#scatter_plot
plt.scatter(x.crime_rate, x.percent_senior)
plt.title('Plot rate of crime vs percent senior')
plt.xlabel('Percent senior')
plt.ylabel('Rate of crime')
plt.show()


# In[34]:


#line_plot
plt.figure(figsize=(12,5))
plt.plot(x.work_force,x.income, linestyle="--",marker='o',color='green')
plt.xlabel('Work Force')
plt.ylabel('Income')
plt.show()


# In[40]:


plt.plot(x.work_force,x.income, '--ro')
plt.xlabel('Work Force')
plt.ylabel('Income')
plt.show()


# In[46]:


plt.plot(x.work_force,x.income, 'gx')
plt.show()


# In[54]:


plt.plot(x.work_force,x.income, color="red",label='Work force')
plt.plot(x.physicians,x.income, label='Physicians')
plt.legend()
plt.show()


# In[72]:


# 1 row and 2 column
plt.subplot(1,2,1)
plt.plot(x.work_force,x.income,"go")
plt.title("Income vs Work Force")

plt.subplot(1,2,2)
plt.plot(x.hospital_beds,x.income,"r^")
plt.title("Income vs Hospital beds")

plt.suptitle('Sub plot')
plt.show()


# In[74]:


# 2 row and 1 column
plt.subplot(2,1,1)
plt.plot(x.work_force,x.income,"go")
plt.title("Income vs Work Force")

plt.subplot(2,1,2)
plt.plot(x.hospital_beds,x.income,"r^")
plt.title("Income vs Hospital beds")

plt.suptitle('Sub plot')
plt.show()


# In[87]:


fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(6,6))
ax[0,0].plot(x.work_force,x.income,"go")
ax[0,1].plot(x.work_force,x.income,"bo")
ax[1,0].plot(x.work_force,x.income,"yo")
ax[1,1].plot(x.work_force,x.income,"ro")
plt.show()


# In[90]:


#Histogram of matplotlap
plt.title('Histogram')
plt.xlabel('Percentage of senior citizens')
plt.ylabel('Frequency')
plt.hist(x.percent_senior)
plt.show()


# In[93]:


import seaborn as sns


# In[97]:


iris = sns.load_dataset('iris')
iris.sample(10)


# In[98]:


sns.scatterplot(x='sepal_length',y="sepal_width", data= iris)
plt.show()


# In[102]:


#swarmplot
sns.swarmplot(x='species',y="petal_length", data= iris)
plt.show()


# In[107]:


#Heatmap
sns.heatmap(iris.corr(),annot=True)
plt.show()


# In[113]:


#Bar chart
#Vertical
divisions = ['A','B','C','D','E']
division_avg = [70,82,73,65,68]

plt.bar(divisions,division_avg,color='green')
plt.title('Bar Graph')
plt.xlabel('Divisions')
plt.ylabel('Marks')
plt.show()


# In[119]:


#Horizontal
divisions = ['A','B','C','D','E']
division_avg = [70,82,73,65,68]
plt.barh(x.region,x.crime_rate,color='green')
# plt.barh(divisions,division_avg,color='green')
plt.xlabel('Divisions')
plt.ylabel('Marks')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




