#!/usr/bin/env python
# coding: utf-8

# # Data Exploratory

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# import csv file 
filepath = r"C:\Users\QXJ\Desktop\IBM/automobile_clean.csv"
df = pd.read_csv(filepath)


# In[ ]:


df.dtypes


# In[11]:


df.head()


# ## Analyzing Individual Feature Patterns Using Visualization

# ### Numerica data 

# In[12]:


# Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
df[["bore","stroke","compression-ratio","horsepower"]].corr()


# In[13]:


# regression plot of "engine-size" and "price"
sns.regplot(x="engine-size", y="price",data=df)
plt.ylim(0,)


# In[14]:


# check correlation between engine size and price
df[["engine-size","price"]].corr()


# In[15]:


# correlation between highway-mpg and price
df[["highway-mpg","price"]].corr()


# In[16]:


# regression plot 
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# In[20]:


# correlation between peak-rpm and price
df[["peak-rpm","price"]].corr()


# In[21]:


sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# In[22]:


# correlation between stroke and price
df[["stroke","price"]].corr()


# In[23]:


sns.regplot(x="stroke", y="price",data = df )
plt.ylim(0,)


# #### Conclusion: price has strong relation to engine size and highway-mpg

# ### Categorical data 

# In[24]:


sns.boxplot(x="body-style", y="price", data=df)


# In[26]:


sns.boxplot(x="engine-location", y="price", data=df)


# In[27]:


sns.boxplot(x="drive-wheels", y="price", data=df)


# ### Descriptive Statistical Analysis
# - the count of that variable
# - the mean
# - the standard deviation (std)
# - the minimum value
# - the IQR (Interquartile Range: 25%, 50% and 75%)
# - the maximum value

# In[29]:


df.describe()


# In[30]:


# count drive-wheels
df["drive-wheels"].value_counts()


# In[33]:


# change to dataframe
df["drive-wheels"].value_counts().to_frame()


# In[35]:


# rename column name
drive_wheel_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheel_counts.rename(columns = {"drive-wheels":"value-counts"}, inplace=True)
print(drive_wheel_counts)


# In[36]:


# rename index name 
drive_wheel_counts.index.name = "drive-wheels"
print(drive_wheel_counts)


# In[40]:


# create engine-location and its counts dataframe 
engine_location_counts = df["engine-location"].value_counts().to_frame()
print(engine_location_counts)


# In[41]:


engine_location_counts.rename(columns={"engine-location":"value-counts"}, inplace=True)
engine_location_counts.index.name = "engine-location"
print(engine_location_counts)


# In[43]:


engine_location_counts.head()
# only 3 in rear can not draw solid conclusion


# ### Basic grouping 

# In[44]:


# return the unique values
df["drive-wheels"].unique()


# In[50]:


df_group_one = df[["drive-wheels", "body-style", "price"]]
df_group_one


# In[51]:


# group the mean of group_one by drive wheels
df_group_one = df_group_one.groupby("drive-wheels", as_index = False).mean()
df_group_one


# In[53]:


# now group both by drive-wheels and body-stlye
df_group_two = df[["drive-wheels", "body-style", "price"]]
df_group_two


# In[55]:


df_group_two = df_group_two.groupby(["drive-wheels", "body-style"], as_index = False).mean()
df_group_two


# In[64]:


# pivot the table 
df_gtwo_pivo = df_group_two.pivot(index = "drive-wheels", columns = "body-style")
df_gtwo_pivo


# In[65]:


# fill in NaN value
df_gtwo_pivot = df_gtwo_pivot.fillna(0)
df_gtwo_pivot


# In[69]:


# find the average "price" of each car based on "body-style".
df_group_body = df[["body-style","price"]]
df_group_body = df_group_body.groupby(["body-style"], as_index = False).mean()
df_group_body


# In[74]:


# plot heatmap
plt.pcolor(df_gtwo_pivot, cmap = "RdBu")
plt.colorbar()
plt.show()


# In[75]:


fig, ax = plt.subplots()
im = ax.pcolor(df_gtwo_pivot, cmap='RdBu')

#label names
row_labels = df_gtwo_pivot.columns.levels[1]
col_labels = df_gtwo_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(df_gtwo_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_gtwo_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# #### conclusion: rwd/convertiable is the most expensive one and 4wd/covertible is the cheapest. 

# ## Correlation and Causation
# 
# #### P-value: the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.
# 
# - p-value is<0.001: we say there is strong evidence that the correlation is significant.
# - the p-value is <0.05: there is moderate evidence that the correlation is significant.
# - the p-value is <0.1: there is weak evidence that the correlation is significant.
# - the p-value is > 0.1: there is no evidence that the correlation is significant.

# In[76]:


df.corr()


# In[77]:


import scipy
from scipy import stats


# In[80]:


# calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'
pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
# Conclusion: since the p-value is  < 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585).


# In[84]:


# calculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'
pearson_coef, p_value = stats.pearsonr(df["length"], df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between length and price is statistically significant, and the linear relationship is moderately strong (~0.691).


# In[85]:


# calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df["width"], df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751).


# In[88]:


# calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df["curb-weight"], df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between curb-weight and price is statistically significant, and the linear relationship is quite strong (~0.834).


# In[89]:


# calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price'
pearson_coef, p_value = stats.pearsonr(df["engine-size"], df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).


# In[90]:


# calculate the Pearson Correlation Coefficient and P-value of 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df["bore"],df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).


# In[92]:


# calculate the Pearson Correlation Coefficient and P-value of "city-mpg" and 'price'
pearson_coef, p_value = stats.pearsonr (df["city-mpg"], df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of about -0.687 shows that the relationship is negative and moderately strong.


# In[93]:


# calculate the Pearson Correlation Coefficient and P-value of 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df["highway-mpg"], df["price"])
print("The pearson correlation coefficient is " + str(pearson_coef) + " with a p-value of p =" + str(p_value))
# Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of about -0.705 shows that the relationship is negative and moderately strong.


# ## ANOVA: Analysis of Variance
# #### To test whether there are significant differences between the means of two or more groups. ANOVA returns two parameters:
# 
# - F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.
# 
# - P-value: P-value tells how statistically significant our calculated score value is.
# 
# #### If our price variable is strongly correlated with the variable we are analyzing, we expect ANOVA to return a sizeable F-test score and a small p-value.

# In[104]:


# group by drive -wheels
df_group_two = df[["drive-wheels","price"]]
group_test2 = df_group_two.groupby(["drive-wheels"], as_index = False)
group_test2.head()


# In[105]:


# obtain the values of the method group using the method "get_group"
group_test2.get_group("4wd")["price"]


# In[107]:


# ANOVA on drive-wheels and price 
f_val, p_val = stats.f_oneway(group_test2.get_group('fwd')['price'], group_test2.get_group('rwd')['price'], group_test2.get_group('4wd')['price'])   
print( "ANOVA results: F=", f_val, ", P =", p_val) 
# This is a great result with a large F-test score showing a strong correlation and a P-value of almost 0 implying almost certain statistical significance.


# In[108]:


# ANOVA on drive-wheels - fwd and rwd and price
f_val, p_val = stats.f_oneway(group_test2.get_group("fwd")["price"], group_test2.get_group("rwd")["price"])
print("ANOVA results: F= "+str(f_val)+", p= "+str(p_val))


# In[109]:


# ANOVA on drive-wheels - 4wd and rwd and price 
f_val, p_val = stats.f_oneway(group_test2.get_group("4wd")["price"], group_test2.get_group("rwd")["price"])
print("ANOVA results: F= "+str(f_val)+", p= "+str(p_val))


# In[110]:


# ANOVA on drive-wheels - 4wd and fwd and price 
f_val, p_val = stats.f_oneway(group_test2.get_group("4wd")["price"], group_test2.get_group("fwd")["price"])
print("ANOVA results: F= "+str(f_val)+", p= "+str(p_val))


# In[111]:


filepath = "C:/Users/QXJ/Desktop/IBM/automobile_explotary.csv"
df.to_csv(filepath)

