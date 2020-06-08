#!/usr/bin/env python
# coding: utf-8

# # Week 1 - Topic 1
# **Authors:** Saravanan JaiChandaran **|** Julián Darío Miranda **|** Manish Kumar Chaudhary
# # `Pandas`, Data Analysis library
# 

# `Pandas` is a data analysis library that provides a variety of data structures and data manipulation methods that allows to perform complex tasks with simple one-line commands.
# 
# In the following sections, we will be working on the `Pandas` library and its uses. We will be reviewing how the objects are defined by `pandas`, operating their values and executing a series of important operations for data analysis. Let us begin!

# ## 1. Installing and Using Pandas

# The downloading and installation of `pandas` library process can be done through the `pip` standard package-management system, by executing in your local environment's console this single line:
# 
# `pip install pandas`
# 
# Once installed, we can import it in the following way:

# In[ ]:


import pandas
pandas.__version__


# It is good practice to create the alias `pd` for `pandas`:

# In[ ]:


import pandas as pd


# ## 2. Reviewing Pandas Objects

# At the core of the `` pandas`` library there are two fundamental data structures/objects:
# 1. **`Series`**: stores single column data along with an **index**. An index is just a way to "number" the `Series` object.
# 2. **`DataFrame`**: is a two-dimensional tabular data structure with labeled axes. It is conceptually useful to think of a `DataFrame` object as a collection of `Series` objects. That is, think of each column in a DataFrame as a single object in the `Series`, where each of these objects in the `Series` shares a common index: the index of the `DataFrame` object.
# 
# Let's import the necessary libraries and we will delve into each of these `pandas` concepts:

# In[151]:


import numpy as np
import pandas as pd


# ### 2.1. Pandas `Series` Object
# 
# 

# A Pandas ``Series`` is a one-dimensional array of indexed data.
# It can be created from a list or array as follows:

# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0])
data


# As we see in the output, the `Series` wraps both a sequence of values and a sequence of indices starting from 0 to the number of values added through the list, which we can access with the `values` and `index` attributes. The values are simply a familiar `NumPy` array:

# In[ ]:


data.values


# The index is an array-like object of type `pd.Index`.

# In[ ]:


data.index


# Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket slicing notation:

# In[ ]:


data[1]


# In[ ]:


data[1:3]


# As we will see, the Pandas ``Series`` is much more general and flexible than the one-dimensional NumPy array that it emulates.

# #### 2.1.1. `Series` as generalized NumPy array

# `Numpy` Array has an implicitly defined integer index used to access the values. The Pandas `Series` has an explicitly defined `index` associated with the values.

# This explicit `index` definition gives the `Series` object additional capabilities. The `index` does not need an integer value mandatory, but can consist of values of any desired type, as we can see in the following example:

# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data


# As we can see, we have defined the indices as characters of the English alphabet. One important fact is that the number of defined indices corresponds to the number of added values in the `Series`. When accessing the values associated with the indexes, we must consider the new index nomenclature and access the values with the common slicing:

# In[ ]:


data['b']


# We can even use non-contiguous or non-sequential indices (index), considering the number of defined indices corresponds to the number of added values in the `Series`:

# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data


# And we access the values by slicing the indices such as:

# In[ ]:


data[5]


# #### 2.1.2 `Series` as specialized dictionaries

# A dictionary is a structure that maps arbitrary keys to a set of arbitrary values, and a `Series` is a structure which maps typed keys to a set of typed values. We can take advantage of these similarities to create a `Series` from a dictionary, where the `keys` are the `indices` of the `Series` and the `values` are those associated with these `indices`.

# In[ ]:


population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population


# From here, typical dictionary-style item access can be performed:

# In[ ]:


population['California']


# Unlike a dictionary, the ``Series`` also supports array-style operations such as slicing:

# In[ ]:


population['California':'Illinois']


# ### 2.2. Pandas `DataFrame` Object

# A ``DataFrame`` is an analog of a two-dimensional array with both flexible row indices and flexible column names.
# Just as you might think of a two-dimensional array as an ordered sequence of aligned one-dimensional columns, you can think of a ``DataFrame`` as a sequence of aligned ``Series`` objects.
# Here, by "aligned" we mean that they share the same index.
# 
# To demonstrate this, let's first construct a new ``Series`` listing the area of each of the five states discussed in the previous section:

# In[ ]:


area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area


# Now that we have this along with the ``population`` Series from before, we can use a dictionary to construct a single two-dimensional object containing this information:

# In[ ]:


states = pd.DataFrame({'population': population,
                       'area': area})
states


# Like the ``Series`` object, the ``DataFrame`` has an ``index`` attribute that gives access to the index labels:

# In[ ]:


states.index


# Additionally, the ``DataFrame`` has a ``columns`` attribute, which is an ``Index`` object holding the column labels:

# In[ ]:


states.columns


# Thus the ``DataFrame`` can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for accessing the data.

# There are some functions and attributes that allow us to observe basic information about the data stored in a `DataFrame` object:
# 
# 1. `DataFrame.head()` -> returns the content of the first 5 rows, by default
# 2. `DataFrame.tail()` -> returns the content of the last 5 rows, by default
# 3. `DataFrame.shape` -> returns a tuple of the form (num_rows, num_columns)
# 4. `DataFrame.columns` -> returns the name of the columns
# 5. `DataFrame.index` -> returns the index of the rows
# 
# Using `` `data.head ()` `` and `` `data.tail ()` `` we can see the content of the data. Unless otherwise specified, `DataFrame` and `Series` objects have indexes starting from 0 and incrementing monotonically and incrementally as integers.

# In[ ]:


states.head(3) # The first three rows


# In[ ]:


states.tail(3) # The last three rows


# ### 2.3. Pandas `Index` Object
# 
# This ``Index`` object is an interesting structure itself, and it can be thought as an *immutable array*:

# In[ ]:


ind = pd.Index([2, 3, 5, 7, 11])
ind


# This index object can be sliced as a list or a numpy array:

# In[ ]:


ind[1] # Accessing element in position 1


# In[ ]:


ind[::2] # Accessing elements starting from position 0 through all the elements two by two.


# Some common attributes with `Numpy` arrays:

# In[ ]:


print(' Size:',ind.size,'\n',
      'Shape:',ind.shape,'\n',
      'Dimension:',ind.ndim,'\n', 
      'Data type:',ind.dtype)


# One difference between ``Index`` objects and `Numpy` arrays is that indices are **immutable**. That is, they cannot be modified via the normal means, which will cause an error:

# In[ ]:


ind[1] = 0


# ## 3. Data Indexing and Selection

# Let's see in detail how to access the elements of the `Series` and `DataFrames` objects.

# ### 3.1. Data Selection in `Series`

# Let's redefine a `Series` object for explanatory purposes:

# In[ ]:


data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data


# In addition to accessing the elements of a `Series` through slicing the indexes, we can update the values associated with them as:

# In[ ]:


data['d'] = 0.95 # updating the value associated with 'd' index in Series object
data


# We can also add a new value using the same procedure, just as we would do with a dictionary:

# In[ ]:


data['e'] = 1.25 # New value added with 'e' index in Series object
data


# Slicing access can be done *explicitly* or *implicitly* through a range, such that:

# In[ ]:


# slicing by explicit index
data['a':'c']


# In[ ]:


# slicing by implicit integer index
data[0:2]


# We can access a group of specific indices by including these indices in a list, in such a way that access is done as follows:

# In[ ]:


# express indexing
data[['a', 'e']]


# Notice that when slicing with an explicit index (i.e., `data ['a':'c']`), the final index is included in the slice, while when slicing with an implicit index (i.e., `data[0:2]`), the final index is excluded from the slice. When we slicing through a list (i.e., `data [['a', 'e']]`), all indices are equally accessed.

# ### 3.2. Indexers: loc, iloc for `Series`

# We are going to review two ways to access the elements of a `Series`, using two attributes determined for this: `.loc[]` and `.iloc[]`. Let's first define a `Series` of three string elements and character indices:

# In[ ]:


data = pd.Series(['Hello', 'DPhi', 'world'], index=['a', 'b', 'c'])
data


# #### 3.2.1. `loc` attribute

# The ``loc`` attribute allows indexing and slicing that always references the explicit index (the explicit name of the index):

# In[ ]:


data.loc['a']


# In[ ]:


data.loc['a':'c']


# #### 3.2.2. `iloc` attribute

# The ``iloc`` attribute allows indexing and slicing that always references the implicit Python-style index (the direct row number of the values monotonically increasing from 0 to 3):

# In[ ]:


data.iloc[1]


# In[ ]:


data.iloc[1:3]


# Exercise 1
# 
# Consider the following lists:
# ```
# lst1 = [1, 2, 3, 5, 8]
# lst2 = [8, 5, 3, 2, 1]
# ```
# 
# 1. Create and display two individual `Series` objects `s1` and `s2` from the data available on each list.
# 
# 
# 2. Perform the following operations with the two series (element-wise):
#     1. Add `s1` and `s2` and store the result in a new variable `s3_add`
#     2. Subtract `s2` from `s1` and store the result in a new variable `s3_sub`
#     3. Multiply `s1` and `s2` and store the result in a new variable `s3_mul`
#     4. Divide `s1` by `s2` and store the result in a new variable `s3_div`

# In[34]:


# Answer 1
s1 = pd.Series([1,2,3,5,8])
s1


# In[29]:


s2 = pd.Series([8, 5, 3, 2, 1])
s2


# In[30]:


# Answer 2
s3_add =s1.add(s2)
s3_add


# In[23]:


s3_sub = s1.sub(s2)
s3_sub


# In[26]:


s3_mul = s1.mul(s2)
s3_mul


# In[27]:


s3_div = s1.div(s2)
s3_div


# In[ ]:


Exercise 2

Consider the following `Series` object:
```
0    45000
1    37872
2    57923
3    68979
4    78934
5    69897
6    56701
Name: Amazon_Reviews, dtype: int64
```

1. Create and display the `Amazon_Reviews` Series.

2. Get the last three values from `Amazon_Reviews` using negative indexing.


# In[40]:


# Answer 1
Amazon_Reviews = pd.Series([4500,37872,57923,68979,78934,69897,56701])
Amazon_Reviews


# In[56]:


## Answer 2
Amazon_Reviews[-4:-1]


# ## Exercise 3
# 
# Consider the following dictionary which is relating the area in sq units of some USA states: 
# ```
#     area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
#              'Florida': 170312, 'Illinois': 149995}
# ```
# 
# 1. Create a `Series` using the given dictionary
# 2. Extract areas for 'Texas', 'New York',  and 'Florida' from the created series

# In[60]:


# Answer 1
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
a = pd.Series(area_dict)
a


# In[63]:


# Answer 2
a[1:4].values


# In[65]:


a['Texas']


# In[67]:


a['New York']


# In[66]:


a['Florida']


# ### 3.3. Data Selection in `DataFrame`

# Let's see in detail how to access the elements of the `DataFrame` objects, redefine a `DataFrame` object for explanatory purposes:

# In[ ]:


area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data


# The individual ``Series`` that make up the columns of the ``DataFrame`` can be accessed via dictionary-style indexing of the column name:

# In[ ]:


data['area'] # Accessing the 'area' column of the data DataFrame


# In[ ]:


data['pop'] # Accessing the 'pop' column of the data DataFrame


# Equivalently, we can use attribute-style access with column names that are strings, which will result in the exact same `Series` output:

# In[ ]:


data.area # Equivalent to data['area']


# Like with the ``Series`` objects discussed earlier, this dictionary-style syntax can also be used to modify the object, in this case adding a new column:

# In[ ]:


data['density'] = data['pop']/data['area']
data


# In[ ]:


As you can see, when the `Series` 'pop' is accessed and divided over the `Series` 'area', the arithmetic operation becomes element-wise and the result is assigned to the new `Series` 'density', which becomes the third column of the `DataFrame` `data`.


# We can also view the ``DataFrame`` as an enhanced two-dimensional array.
# We can examine the raw underlying data array using the ``values`` attribute, which will return a two-dimensional array in which each row corresponds to a row of `DataFrame` values:

# In[ ]:


data.values


# ### 3.4. Indexers: loc, iloc for `DataFrame`

# We are going to review two ways to access the elements of a `DataFrame`, using two attributes determined for this: `.loc[]` and `.iloc[]`:

# #### 3.4.1. `loc` attribute

# The ``loc`` attribute allows indexing and slicing that always references the explicit index (the explicit name of the index):

# In[ ]:


data.loc[:'Illinois', :'pop']


# In this example we are slicing on the indices starting with the first by default and ending in 'Illinois', as well as by the columns starting with the first by default, and ending in 'pop'.

# #### 3.4.2. `iloc` attribute

# The ``iloc`` attribute allows indexing and slicing that always references the implicit Python-style index (the direct row number of the values monotonically increasing from 0 to 3):

# In[ ]:


data.iloc[:3, :2]


# In this example we are slicing the `DataFrame` from index 0 by default to 3 exlusive, and from column 0 by default to 2 exlusive.

# In[ ]:


Exercise 4

Consider below DPhi Bootcamp's information about different batches:

```
Total_Candidates = {'absolute_beginners': 785, 'beginners': 825, 'intermediat_advanced': 602} # this is true data
Active_Candidates = {'absolute_beginners': 500, 'beginners': 425, 'intermediat_advanced': 300}  # this is hypothetical data
```
   
1. Create a Pandas `DataFrame` using above information (name your Dataframe as `DPhi`)
2. Get all the columns in DPhi.
3. Get the information of total candidates present in each batches using dictionary-style indexing.
4. Find the number of candidates for each batches who are not active and add this information to the dataframe DPhi.
5. Also, find the percent of candidates that are active in each batches and add this information to the `DPhi` dataframe (hint: $percent = (active / total)* 100$)
6. Get all the batches where percentage of active candidates are greater than 60%


# In[84]:


# Answer 1
Total_Candidates = {'absolute_beginners': 785, 'beginners': 825, 'intermediat_advanced': 602} # this is true data
Active_Candidates = {'absolute_beginners': 500, 'beginners': 425, 'intermediat_advanced': 300}

DPhi = pd.DataFrame({'Total_Candidates':Total_Candidates,'Active_Candidates':Active_Candidates})
DPhi


# In[96]:


# Answer 2
DPhi.columns


# In[98]:


# Answer 3
DPhi.Total_Candidates


# In[102]:


# Answer 4
DPhi['Non_Active'] = DPhi['Total_Candidates'] - DPhi['Active_Candidates']
DPhi


# In[103]:


# Answer 5
DPhi['Active_in_Percent'] = (DPhi['Active_Candidates'] / DPhi['Total_Candidates'])*100
DPhi


# In[107]:


# Answer 6
DPhi[DPhi.Active_in_Percent > 60]


# ### 3.5. Subsetting a `Dataframe`

# **Subsetting** a `DataFrame` is a way of filtering which allows to extract portions of interest. Subsetting can be done using comparison operators and logical operators inside a pair of square brackets `[]` as shown in the following example:

# In[ ]:


data[data['density'] > 100]


# In the above example, we are extracting the rows for which it is `True` that the population density is greater than 100 units. For more clarification on the logical operation concept, take a look to the following extract of the above example:

# In[ ]:


data['density'] > 100


# Since there are only two rows where the density is greater than 100, the result will be a boolean-values `DataFrame` in which the `True` values correspond to the rows that accomplish the locigal expression, and the `False` values to the ones that do not.
# 
# Let's see an example in which we include a second logical operation to select those rows in which the population density is greater than 100 units `data['density'] > 100` and (`&`) the area is less than 150,000 units `data['area'] < 150000`:

# In[ ]:


data[(data['density'] > 100) & (data['area'] < 150000)]


# We could also select those records in which the population density is less than 90 units `data['density'] < 90` or (`|`) greater than 120 `data['density'] > 120`:

# In[ ]:


data[(data['density'] < 90) | (data['density'] > 120)]


# The previous example can be rewritten to express the negation of the above condition, select those records that do not have a population density greater than or equal to 90 units and less than or equal to 120 units `~((data['density'] >= 90) & (data['density'] <= 120))`:

# In[ ]:


data[~((data['density'] >= 90) & (data['density'] <= 120))]


# As we can see, the results of the last two examples are the same, since we are expressing the same condition in different ways.

# ## 4. Data Wrangling

# The difference between data found in many tutorials and data from the real world is that real-world data is rarely clean and homogeneous. In particular, many interesting datasets will have some amount of data missing. To make matters even more complicated, different data sources may indicate missing data in different ways. 
# 
# In this way, we need to define methods that allow us to structure, clean and enrich the data acquired from the real world, which are the main steps for **Data Wrangling**. Before continuing, let's see what is the difference between these three steps and expand their definition:
# 
# **1. Data structuring:** 
# 
# The first step in the data wrangling process is to separate the relevant data into multiple columns, so that the analysis can be run grouping by common values in a separate way. In turn, if there are columns that are not desired or that will not be relevant to the analysis, this is the phase to filter the data or mix together some of their columns.
# 
# **2. Data Cleaning**
# 
# In this step, the data is cleaned up for high-quality analysis. `Null values` are handled, and the data format is standardized. We will enter this process in the following weeks.
# 
# **3. Data Enriching**
# 
# After cleaning, the data is enriched by increasing some variables in what is known as *Data Augmentation* and using additional sources to enrich them for the following stages of processing.
# 
# For now, we will review how to handle missing values, a fundamental step for data cleaning.

# ## 5. Handling Missing Data

# This is a fundamental step in data cleaning. It is common that during the data acquisition processes, there are lost records, either due to the difficulties of acquiring them, due to errors in the source or destination, or because we simply could not acquire the data. There are three types of missing data:
# 
# - Missing completely at random (MCAR): when the fact that the data is missing is independent of the observed and unobserved data.
# - Missing at random (MAR): when the fact that the data is missing is systematically related to the observed but not the unobserved data.
# - Missing not at random (MNAR): when the missingness of data is related to events or factors which are not measured by the researcher.
# 
# We will go into these types in detail later. For now, we'll look at the fundamentals of handling missing data in pandas:

# ### 5.1. `NaN` and `None` in Pandas
# 
# Missing data is handled in Pandas as `NaN` values placeholders. `NaN` value is a IEEE 754 floating point representation of *Not a Number (NaN)*. One of the main reasons to handle missing data as `NaN` rather than `Null` in Pandas is that `NaN` (from `np.nan`) allows for vectorized operations, since it is a float value. `None`, by definition, forces object type, which basically disables all efficiency in Numpy and Pandas.
# 
# ``NaN`` and ``None`` are handled nearly interchangeably by `Pandas`, converting between them where appropriate:

# In[ ]:


import numpy as np

pd.Series([1, np.nan, 2, None])


# ### 5.2. Operations on Missing Values
# 
# There are several useful methods for detecting, removing, and replacing missing values in Pandas data structures:
# 
# - ``isnull()``: generates a boolean mask indicating missing values
# - ``notnull()``: generates a boolean mask of non-missing values. Is the opposite of ``isnull()``.
# - ``dropna()``: returns a filtered version of the data, without missing values.
# - ``fillna()``: returns a copy of the data with missing values filled or imputed with a desired strategy.
# 
# Let's review some examples of the first two functions `isnull()` and `notnull()`:

# In[ ]:


data = pd.Series([1, np.nan, 'hello', None])
data


# In[ ]:


data.isnull()


# In[ ]:


data.notnull()


# As you can see, the `.isnull()` function returns a Dataframe with boolean values, where `False` denotes a present value and `True` denotes a missing value. Conversely, the `.notnull()` function returns a Dataframe with boolean values, where `True` denotes a present value and `False` denotes a missing value.
# 
# With this boolean result we can make a subsetting to filter those missing values:

# In[ ]:


data[data.notnull()]


# In[ ]:


data


# Although we have filtered the missing data, if we review the content of the variable again, we will see that the missing data persists. This is because no subsetting operation is done inplace. Now let's see how we remove missing data from our dataset.

# ### 5.3. Dropping missing values
# 
# The basic function to remove any missing values from a `Series` object is as follows, although the function is not executed inplace:

# In[ ]:


data.dropna()


# In[ ]:


data


# For a ``DataFrame``, there are more options.
# Consider the following ``DataFrame``:

# In[ ]:


df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df


# We cannot drop single values from a ``DataFrame``; we can only drop full rows or full columns.
# Depending on the application, you might want one or the other, so ``dropna()`` gives a number of options for a ``DataFrame``.
# 
# By default, ``dropna()`` will drop all rows in which *any* missing value is present:

# In[ ]:


df.dropna()


# Alternatively, you can drop missing values along a different axis; `axis=1` drops all columns containing a missing value:

# In[ ]:


df.dropna(axis='columns')


# But this drops some good data as well; you might rather be interested in dropping rows or columns with *all* `NaN` values, or a majority of `NaN` values. This can be specified through the ``how`` or ``thresh`` parameters, which allow fine control of the number of nulls to allow through.
# 
# The default is ``how='any'``, such that any row or column (depending on the ``axis`` keyword) containing a null value will be dropped. You can also specify ``how='all'``, which will only drop rows/columns that are *all* null values:

# In[ ]:


df = pd.DataFrame([[1,      np.nan, 2, np.nan],
                   [2,      3,      5, np.nan],
                   [np.nan, 4,      6, np.nan]])
df


# In[ ]:


df.dropna(axis='columns', how='all')


# ### 5.4. Filling null values
# 
# Sometimes rather than dropping `NaN` values, you'd rather replace them with a valid value.
# This value might be a single number like zero, or it might be some sort of imputation or interpolation from the good values.
# 
# There are four types of treatment that can be given, in that order, to unwanted non-existent or missing data:
# 
# 1. **Treatment 1:** Ignore the missing or unwanted data in some columns, considering that in other columns of the same rows there are important or relevant data for the study.
# 2. **Treatment 2:** Replace the missing or unwanted data with values that represent an indicator of nullity.
# 3. **Treatment 3:** Replace the missing, nonexistent or unwanted data with interpolated values that are related to the trend of the data that is present.
# 4. **Treatment 4:** Delete the missing data, with the certainty that valuable information will not be lost when analyzing the data.
# 
# You can apply **Treatment 2** and **Treatment 3** in-place using the ``isnull()`` method as a mask, but because it is such a common operation `Pandas` provides the ``fillna()`` method, which returns a copy of the array with the missing values replaced.
# 
# Consider the following ``Series``:

# In[ ]:


data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data


# To replace the missing values of the Series with a null value, we can do the following:

# In[ ]:


data.fillna(0)


# We can specify a forward-fill to propagate the previous value forward:

# In[ ]:


# forward-fill
data.fillna(method='ffill')


# Or we can specify a back-fill to propagate the next values backward:

# In[ ]:


# back-fill
data.fillna(method='bfill')


# For ``DataFrame`` objects the options are similar, but we can also specify an ``axis`` along which the fills take place:

# In[ ]:


df


# In[ ]:


df.fillna(method='ffill', axis=1) # forward-fill along the columns


# In[ ]:


df.fillna(method='ffill', axis=0) # forward-fill along the rows


# Notice that if a previous value is not available during a forward fill, the `NaN` value remains.

# ## 6. `Pandas` String Operations

# When a `Pandas` object stores string data, `Pandas` provides certain operations to facilitate its manipulation. Let's see what would happen if a classic data storage structure like a list had missing data and a string operation was executed. Firstly, we define a list with four string values:

# In[ ]:


data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]


# This is perhaps sufficient to work with some data, but it will break if there are any missing values:

# In[ ]:


data = ['peter', 'Paul', None, 'MARY', 'gUIDO'] #this has a missing value None hence the error below
[s.capitalize() for s in data]


# Now let's look at a Pandas `Series`:

# In[ ]:


import pandas as pd
names = pd.Series(data)
names


# We can now call a single method that will capitalize all the entries, while skipping over any missing values or non-string values:

# In[ ]:


names.str.capitalize()


# We have accessed the `str` attribute that parses the values stored in the `Series` to string.

# ### 6.1. String Methods

# Here is a list of Pandas ``str`` methods that mirror Python string methods:
# 
# |             |                  |                  |                  |
# |-------------|------------------|------------------|------------------|
# |``len()``    | ``lower()``      | ``translate()``  | ``islower()``    | 
# |``ljust()``  | ``upper()``      | ``startswith()`` | ``isupper()``    | 
# |``rjust()``  | ``find()``       | ``endswith()``   | ``isnumeric()``  | 
# |``center()`` | ``rfind()``      | ``isalnum()``    | ``isdecimal()``  | 
# |``zfill()``  | ``index()``      | ``isalpha()``    | ``split()``      | 
# |``strip()``  | ``rindex()``     | ``isdigit()``    | ``rsplit()``     | 
# |``rstrip()`` | ``capitalize()`` | ``isspace()``    | ``partition()``  | 
# |``lstrip()`` |  ``swapcase()``  |  ``istitle()``   | ``rpartition()`` |

# Let's see some examples of string methods for Pandas `Series` with the `monte` Series:

# In[ ]:


monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])


# In[ ]:


monte.str.lower() # Parse values to string and transform all characters to lowercase


# In[ ]:


monte.str.len() # Parse values to string and calculates their length


# In[ ]:


# Parse values to string and calculates a mask of string values starting by 'T'
monte.str.startswith('T')


# In[ ]:


monte.str.split() # Parse values to string and splits them by ' ' character, by default


# In[ ]:


## Exercise 5

Consider the following lists:

```
country = ['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar']
year = [2002, 2002, 1957, 2007, 1967]
population = [16122830.0, np.nan, 9146100.0, 6426679.0, 6334556.0]
continent = ['Europe', 'europe', 'Americas', 'asia', 'Africa']
```

1. Create a Dataframe object which contains all the lists values as Series. The final DataFrame should be named as `country_info`, containing 4 columns and 5 rows.
2. Delete the rows which contains missing values
3. Capitalize all the continents in continent column.
4. Get the length of each country's names.


# In[126]:


# Answer 1
country = ['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar']
year = [2002, 2002, 1957, 2007, 1967]
population = [16122830.0, np.nan, 9146100.0, 6426679.0, 6334556.0]
continent = ['Europe', 'europe', 'Americas', 'asia', 'Africa']
country_info = pd.DataFrame({'country':country,'year':year,'population':population,'continent':continent})
country_info


# In[121]:


# Answer 2
country_info.dropna(axis='rows')


# In[143]:


# Answer 3
continent = ['Europe', 'europe', 'Americas', 'asia', 'Africa']
[s.capitalize() for s in continent]


# In[142]:


# Answer 4
country = pd.Series(['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar'])
country.str.len()


# ## 7. Concatenate `Series`
# 
# Here we'll take a look at simple concatenation of `Series` and `DataFrame` objects with the `pd.concat()` function:

# In[145]:


ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2], axis=0)


# In[147]:


pd.concat([ser1, ser2], axis=1)


# By default, the concatenation takes place row-wise within the ``DataFrame`` (i.e., ``axis=0``). However, as you can see, the concatenation of `Series` in a `DataFrame` can be done in contiguous rows or columns by specifying the `axis` parameter. In the case where they are columns, care must be taken to define the same index values, so that the columns are placed contiguously without `NaN` values.

# In[ ]:


Exercise 6

Consider the following lists:

```
country = ['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar']
gdp_per_cap = [33724.757780, 30035.801980, 4245.256698, 25523.277100, 1634.047282]
```

1. Create a Dataframe object which contains all the lists values as Series. The final DataFrame should be named as `country_gdp`, containing 2 columns and 5 rows.
2. Concatenate the two dataframes: `country_info` and `country_gdp` with `axis=0` and name it `concat_data`
3. Check if there are any null values in `concat_data`
4. Find total numer of missing values in each column. *hint: Use `.isnull()` and `.sum()` functions*


# In[153]:


# Answer 1
country = ['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar']
gdp_per_cap = [33724.757780, 30035.801980, 4245.256698, 25523.277100, 1634.047282]
country_gdp = pd.DataFrame({'country':country,'gdp_per_cap':gdp_per_cap})
country_gdp 


# In[163]:



# Answer 2
country = pd.Series(['Netherland', 'Germany', 'Peru', 'Israel', 'Madagascar'], index=[0,1,2,3,4])
gdp_per_cap = pd.Series([33724.757780, 30035.801980, 4245.256698, 25523.277100, 1634.047282], index=[5,6,7,8,9])
concat_data =  pd.concat([country, gdp_per_cap], axis=0)
concat_data


# In[166]:


# Answer 3
concat_data.isnull()


# In[174]:


# Answer 4
gdp_per_cap = pd.Series([33724.757780, 30035.801980, 4245.256698, 25523.277100, 1634.047282], index=[5,6,7,8,9])

gdp_per_cap[gdp_per_cap.notnull()]


# ## 8. `DataFrame` fancy table printing

# In the next two cells we are going to define a fancy way to visualize the data of multiple `DataFrame` objects. Let's first use the `IPython.display` library, which allows us to view the contents of `DataFrame` objects individually, in a table fancy way:

# In[ ]:


from IPython.display import display, HTML

display(pd.concat([ser1, ser2], axis=1))


# Now let's look at a function defined by us, which allows us to evidence the data of multiple `DataFrame` objects. It is not necessary for now to have a complete understanding of the following function. What is important at this time is knowing how to use it.

# In[ ]:


class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# Let's also define a user function that allows us to create a `DataFrame` quickly using *Dictionary Comprehension*. This function transforms an input string into a square array that is later converted to a `DataFrame` where the column names are each of the characters in the string:

# In[ ]:


def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

# example DataFrame
make_df('ABC', range(3))


# The function also allows us to concatenate higher-dimensional objects, such as ``DataFrame`` objects:

# In[ ]:


df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')


# In[ ]:


df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis=1)")


# ## Conclusions

# We have learned the fundamentals of the `pandas` library for data analysis, which allows us to execute operations with stored data efficiently, add records and handle missing data. We also highlighted some of the most important functions of Pandas' `Series` and `DataFrame` objects, as well as following a Data Wrangling procedure.
# 
# In the next case study, we will review the fundamentals of data visualization libraries, so that we can identify visual patterns in our data.
