#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[3]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"


# In[4]:


df = pd.read_csv(url)


# #### Get basic information about the dataset 

# In[17]:


df.head()


# In[18]:


df = pd.read_csv(url, header = None)
df.head()


# In[19]:


# replace default header 
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style","drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",  "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers 
df.head()


# In[20]:


df.head()


# In[21]:


df.tail()


# In[22]:


path = "C:/Users/QXJ/Desktop/IBM/automobile.csv"
df.to_csv(path)


# In[23]:


# check data type
df.dtypes


# In[24]:


# check statistics summary 
df.describe()


# In[25]:


# check all the objects
df.describe(include = 'all')


# In[26]:


# check basic insights 
df.info()


# #### Pre-processing data 

# In[27]:


# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head()


# ## Inspecting the missing data 

# In[28]:


# check the missing data 
missing_data = df.isnull()
missing_data.head()


# In[80]:


# using loop to count missing data 
for col in missing_data.columns.values.tolist():
    print(col)
    print(missing_data[col].value_counts())


# - "normalized-losses": 41 missing data -> replace by mean
# - "num-of-doors": 2 missing data -> replace with four 
# - "bore": 4 missing data -> replace by mean
# - "stroke" : 4 missing data -> replace by mean
# - "horsepower": 2 missing data -> replace by mean
# - "peak-rpm": 2 missing data -> replace by mean
# - "price": 4 missing data -> delete 

# ## Dealing with missing data 
# - Drop data
# a. Drop the whole row
# b. Drop the whole column
# - Replace data
# a. Replace it by mean
# b. Replace it by frequency
# c. Replace it based on other functions

# In[50]:


# check missing data in normalized - losses
df["normalized-losses"].isnull().sum()


# In[55]:


# replace the missing values with mean 
normalized = df["normalized-losses"].astype('float')
avg_normalized = normalized.mean()
print(avg_normalized)


# In[56]:


# replace missing value with mean 
df["normalized-losses"].replace(np.nan, avg_normalized,inplace = True)


# In[57]:


# check missing data in normalized - losses
df["normalized-losses"].isnull().sum()


# In[58]:


# Count the num-of-door
df["num-of-doors"].value_counts()


# In[59]:


# replace "four" to the missing data in the col of num-of-door 
df["num-of-doors"].replace(np.nan,"four", inplace = True)


# In[60]:


# count the missing date in num-of-doors
df["num-of-doors"].isnull().sum()


# In[62]:


# replace missing value with average in bored 
avg_bored = df["bore"].astype('float').mean()
df["bore"].replace(np.nan, avg_bored, inplace = True)


# In[63]:


df["bore"].isnull().sum()


# In[64]:


# replace missing value with average in stroke
avg_stroke = df["stroke"].astype('float').mean()
df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# In[65]:


# check missing value again
df["stroke"].isnull().sum()


# In[67]:


# replace missing value with average in horsepower
avg_horsepower = df["horsepower"].astype('float').mean()
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)


# In[68]:


df["horsepower"].isnull().sum()


# In[69]:


#  replace missing value with average in peak-rpm
avg_peak_rpm = df["peak-rpm"].astype('float').mean()
df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace = True)


# In[70]:


df["peak-rpm"].isnull().sum()


# In[71]:


# dealing with the missing values: drop, avearge, medium or mode ? ? 
# drop the rows without prices 
df["price"].astype('float')
df.dropna(subset = ["price"], axis=0, inplace = True)
df.head()


# In[72]:


df["price"].isnull().sum()


# In[74]:


# check the missing data 
missing_data = df.isnull()
# using loop to count missing data 
for col in missing_data.columns.values.tolist():
    print(col)
    print(missing_data[col].value_counts())


# In[76]:


df.dtypes


# ## Data Standardization

# In[77]:


# transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".
df["highway-L/100km"] = 235/df["highway-mpg"]
df.head()


# ## Data Normalization
# #### Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance is 1, or scaling the variable so the variable values range from 0 to 1.

# In[80]:


#  replace original value by (original value)/(maximum value)
df["Norm_height"]=df["height"]/df["height"].max()
df.head()


# ## Binning
# #### Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.

# In[81]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[98]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"],bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[101]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)


# In[102]:


group_names = ['Low', 'Medium', 'High']
# determine what each value of df['horsepower'] belongs to
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(10)


# In[103]:


df["horsepower-binned"].value_counts()


# In[104]:


plt.pyplot.hist(df["horsepower-binned"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# ## Indicator Variable (or Dummy Variable)
# #### a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning, so we can use categorical variables for regression analysis in the later modules.

# In[106]:


df.columns


# In[107]:


# create dummy item for fuel type
dummy_v1 = pd.get_dummies(df["fuel-type"])
dummy_v1.head()


# In[109]:


# rename columns of dummies 
dummy_v1.rename(columns = {"diesel":"fuel-type-diesel","gas":"gas-type-diesel"}, inplace = True)
dummy_v1.head()


# In[110]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df,dummy_v1],axis = 1 )
# drop original column "fuel-type" from "df"
df.drop("fuel-type",axis = 1, inplace = True)
df.head()


# In[111]:


df["aspiration"].value_counts()


# In[113]:


dummy_v2 = pd.get_dummies(df["aspiration"])
dummy_v2.head()


# In[116]:


dummy_v2.rename(columns = {"std":"aspiration-std","turbo":"aspiration-turbo"}, inplace = True)
dummy_v2.head()


# In[136]:


# Merge to df 
df = pd.concat([df,dummy_v2],axis =1)
# drop aspiration column 
df.drop("aspiration", axis = 1, inplace = True)
df.head()


# In[138]:


path = "C:/Users/QXJ/Desktop/IBM/automobile_clean.csv"
df.to_csv(path)


# In[ ]:




