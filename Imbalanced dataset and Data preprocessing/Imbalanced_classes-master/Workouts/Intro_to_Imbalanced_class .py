#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[237]:


import pandas as pd                  # A fundamental package for linear algebra and multidimensional arrays
import numpy as np                   # Data analysis and data manipulating tool
import random                        # Library to generate random numbers
from collections import Counter      # Collection is a Python module that implements specialized container datatypes providing 
                                     # alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.
                                     # Counter is a dict subclass for counting hashable objects
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings in the notebook
import warnings
warnings.filterwarnings("ignore")


# In[62]:


fraud_data = pd.read_csv(r'D:\hakk study\Data Science\Imbalanced dataset\Imbalanced_classes-master\fraud_data.csv')


# In[65]:


fraud_data.head()


# In[ ]:


fraud_data.info()


# In[ ]:


fraud_data.isFraud.value_counts().head(20)


# In[ ]:


fraud_data.isFraud.value_counts()/ len(fraud_data) * 100


# In[ ]:


sns.countplot(fraud_data.isFraud)


# In[88]:


#Missing value in percentage
fraud_data.isnull().sum()/ len(fraud_data) * 100


# In[99]:


#Eliminating column which has more than 20% missing value

fraud_data = fraud_data[fraud_data.columns[fraud_data.isnull().mean() < 0.2]]


# In[101]:


fraud_data.isnull().sum()/ len(fraud_data) * 100


# In[115]:


num_cols = fraud_data.select_dtypes(include=np.number).columns      # getting all the numerical columns
fraud_data.select_dtypes


# In[106]:


num_cols


# In[135]:


fraud_data[num_cols] = fraud_data[num_cols].fillna(fraud_data[num_cols].mean().iloc[0]) # fills the missing values with mean


# In[136]:


#Getting all Categorical column

cat_cols = fraud_data.select_dtypes(include ='object').columns
cat_cols


# In[165]:


fraud_data.isnull().sum()/ len(fraud_data) * 100


# In[166]:


#get_dummies()
fraud_data = pd.get_dummies(fraud_data, columns=cat_cols)


# In[167]:


fraud_data.C1


# In[168]:


fraud_data.head()


# In[173]:


# Separate input features and output feature

X = fraud_data.drop(columns = ['isFraud'])  #input features
Y = fraud_data.isFraud  #output features


# In[177]:


fraud_data.TransactionAmt.max()


# In[183]:


#Standardization-/-Normalization
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(X)
scaled_features = pd.DataFrame(data = scaled_features)
scaled_features.columns = X.columns
scaled_features.head()


# In[187]:


#Splitting the data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 42)

# X_train: independent feature data for training the model
# Y_train: dependent feature data for training the model
# X_test: independent feature data for testing the model; will be used to predict the target values
# Y_test: original target values of X_test; We will compare this values with our predicted values.
 
# test_size = 0.3: 30% of the data will go for test set and 70% of the data will go for train set


# In[190]:


#Dealing with Imbalanced data
#Resampling Techniques - Oversample Minority class

from sklearn.utils import resample


# In[193]:


train_data = pd.concat([X_train,Y_train], axis=1)
train_data


# In[203]:


not_fraud = train_data[train_data.isFraud == 0]
fraud = train_data[train_data.isFraud == 1]
fraud


# In[213]:


fraud_upsampled = resample(fraud, replace = True, n_samples = len(not_fraud), random_state = 27 )
fraud_upsampled


# In[214]:


upsampled = pd.concat([not_fraud,fraud_upsampled])
upsampled.isFraud.value_counts()


# In[ ]:


#Resampling Technique - UnderSample majority class


# In[229]:


not_fraud_downsampled = resample(not_fraud, replace = False, n_samples = len(fraud), random_state = 27)
not_fraud_downsampled


# In[249]:


downsampled = pd.concat([not_fraud_downsampled, fraud])
downsampled.isFraud.value_counts()


# In[278]:


pip install -U imbalanced-learn


# In[281]:


#Generate Synthetic Samples
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 25, ratio = 1.0)
    


# In[283]:


X_train, Y_train = sm.fit_sample(X_train, Y_train)


# In[282]:


np.unique(Y_train,return_counts=True)


# #### Conclusion
# That's it for this notebook. We learned handling missing values, one hot encoding, standardization / normalization, what is imbalanced class and three techniques to deal with imbalanced classes.

# In[ ]:




