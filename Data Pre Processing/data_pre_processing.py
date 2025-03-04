# -*- coding: utf-8 -*-
"""Data Pre-Processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e_yP1Gj_3CQe9cr5MIk-wg4J3WaROF9e

# Introduction


On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

In the Hollywood blockbuster that was modelled on this tragedy, it seemed to be the case that upper-class people, women and children were more likely to survive than others. But did these properties (socio-economic status, sex and age) really influence one's survival chances? 

Based on data of a subset of 891 passengers on the Titanic, I will make a model that can be used to predict survival of other Titanic passengers. 

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

### Outline

- Preprocessing/cleaning of the provided data

# Dataset Download:

train: https://docs.google.com/spreadsheets/d/1hFOPnxVT9fyT4TFlwuGGbDLfclY43P48UV24PNfAW2M/edit?usp=sharing
"""

# Only for Google Colab users
from google.colab import files
uploaded = files.upload()



"""### Preprocessing
First, let's load the training data to see what we're dealing with. We will import the file to a pandas DataFrame:

## Loading Libraries
"""

import pandas as pd
import numpy as np

"""## Loading Data"""

train_data = pd.read_csv('train - train_titanic dataset.csv')
train_data1 = pd.read_csv('train - train_titanic dataset.csv')

train_data.info()

"""Now, let's take a look at the first few rows of the DataFrame:"""

train_data.head(3)

"""# A bit about the dataset

'Pclass' column contains a number which indicates class of the passenger's ticket:  1 for first class, 2 for second class and 3 for third class. 

This could function as a proxy for the socio-economic status of the passenger ('upper', 'middle', 'low'). 


The 'SibSp' column contains the number of siblings + spouses of the passenger also aboard the Titanic;

the 'ParCh' column indicates the number of parents + children of the passenger also aboard the Titanic. 

The 'Ticket' column contains the ticket numbers of passengers (which are not likely to have any predictive power regarding survival);

'Cabin' contains the cabin number of the passenger, if he/she had a cabin, and lastly, 

'Embarked' indicates the port of embarkation of the passenger: **C**herbourg, **Q**ueenstown or **S**outhampton. The meaning of the other columns is clear, I think.

Let's check some more info on the DataFrame:
"""

train_data.info()

"""The DataFrame contains 891 entries in total, with 12 features. Of those 12 features, 10 have non-null values for every entry, and 2 do not: 'Age', which has 714 non-null entries, and 'Cabin', which has only 204 non-null entries (of course, not everyone had a cabin).

If you carefully observe the above summary of pandas, there are total 891 rows, Age shows only 714 (means missing), Embarked (2 missing) and Cabin missing a lot as well. Object data types are non-numeric so we have to find a way to encode them to numerical values.

# **Dropping Columns which are not useful**

Lets try to drop some of the columns which many not contribute much to our machine learning model such as Name, Ticket, Cabin etc.
"""

cols = ['Name', 'Ticket', 'Cabin']
train_data = train_data.drop(cols, axis=1)
train_data1 = train_data1.drop(cols, axis=1)

train_data.info()

"""# Dropping rows having missing values
Next if we want we can drop all rows in the data that has missing values (NaN). You can do it like
"""

train_data = train_data.dropna()

train_data.info()

"""# Problem with dropping rows having missing values

1.   List item
2.   List item


After dropping rows with missing values we find that the dataset is reduced to 712 rows from 891, which means we are wasting data. Machine learning models need data for training to perform well. So we preserve the data and make use of it as much as we can. We will see it later.
Creating Dummy Variables
Now we convert the Pclass, Sex, Embarked to columns in pandas and drop them after conversion.
"""

dummies = []

cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(train_data1[col]))
titanic_dummies = pd.concat(dummies, axis=1)

"""# And finally we concatenate to the original dataframe column wise"""

train_data1 = pd.concat((train_data1,titanic_dummies), axis=1)

"""Now that we converted Pclass, Sex, Embarked values into columns, we drop the redundant same columns from the dataframe"""

train_data1 = train_data1.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

train_data1.info()

train_data1.head(3) #let's the overview of data now after creating dummy variables

"""# Taking Care of Missing Data
All is good, except age which has lots of missing values. Lets compute a median or interpolate() all the ages and fill those missing age values. Pandas has a interpolate() function that will replace all the missing NaNs to interpolated values.
"""

train_data1['Age'] = train_data1['Age'].interpolate()

"""Now lets observe the data columns. Notice age which is interpolated now with imputed new values."""

train_data1.info()

"""# Converting the dataframe to numpy

Now that we have converted all the data to numeric, its time for preparing the data for machine learning models. 

This is where scikit and numpy come into play:
X = Input set with 14 attributes
y = Small y Output, in this case ‘Survived’
Now we convert our dataframe from pandas to numpy and we assign input and output
"""

X = train_data1.values
y = train_data1['Survived'].values

"""X has still Survived values in it, which should not be there. So we drop in numpy column which is the 1st column."""

X = np.delete(X, 1, axis=1)

"""# Dividing data set into training set and test set
Now that we are ready with X and y, lets split the dataset for 70% Training and 30% test set using scikit model_selection
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

