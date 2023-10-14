#!/usr/bin/env python
# coding: utf-8

# In[149]:


#!pip install pytest-warnings
#!pip freeze
#!pip install numpy
#!pip install pandas
#!pip install seaborn
#!pip install matplotlib
#!pip install scikit-learn
#!pip install statsmodels


# In[52]:


import warnings
warnings.filterwarnings("ignore")


# In[53]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# ## Step 1: Import and Inspect Dataset

# In[54]:


hosdata = pd.read_csv("train.csv")
hosdata.head()


# In[55]:


hosdata.shape


# In[56]:


hosdata.isnull().sum()


# In[57]:


hosdata.describe()


# In[58]:


hosdata.describe(include = 'all')


# In[59]:


hosdata.info()


# In[60]:


hosdata.isnull().sum()/hosdata.shape[0]*100


# ## Data Cleaning

# In[61]:


hosdata.columns


# In[62]:


cols = [
       'Alley', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'FireplaceQu', 'GarageType', 
        'GarageFinish', 'GarageQual',
       'GarageCond', 'PoolQC',
       'Fence', 'MiscFeature']

for i in cols:
    
    hosdata[i].fillna('None', inplace = True)


# In[63]:


hosdata.info()


# In[64]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Check if the target variable is normally distributed

# In[65]:


plt.figure(figsize = [6,6])
sns.distplot(hosdata['SalePrice'])


# ##### We see that the target variable 'salePrice' is right skewed

# In[66]:


print('Skewness: ', hosdata['SalePrice'].skew())
print('Kurtosis: ', hosdata['SalePrice'].kurt())


#  ##### Observation: Target variable has skewness greater than 1 and has high density around saleprice of 160000
#  
# ##### Hence, we can do data transformation for this variable.

# In[67]:


#Log Transformation

hosdata['SalePrice'] = np.log(hosdata['SalePrice'])


# In[68]:


#Checking distribution of target variable after log transformation

plt.figure(figsize = [6,6])
sns.distplot(hosdata['SalePrice'])


# ##### We now see target variable is normally distributed and the skewness and kurtosis is also reduced.

# - Drop ID column
# - Convert 'MSSubclass', 'OverallQual', 'OverallCond' to object datatype
# - Convert 'LotFrontage', 'MasVnArea' to numeric datatype from float64

# In[69]:


hosdata.drop('Id', axis = 1, inplace = True)


# In[70]:


hosdata[['MSSubClass', 'OverallQual', 'OverallCond']] = hosdata[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')


# In[71]:


hosdata['LotFrontage'] = pd.to_numeric(hosdata['LotFrontage'], errors = 'coerce')
hosdata['MasVnrArea'] = pd.to_numeric(hosdata['MasVnrArea'], errors = 'coerce')


# In[72]:


hosdata.info()


# In[78]:


null_cols = hosdata.columns[hosdata.isnull().any()]

for i in null_cols:
    if hosdata[i].dtype == np.float64 or hosdata[i].dtype == np.int64:
        hosdata[i].fillna(hosdata[i].mean(), inplace = True )
    else:
        hosdata[i].fillna(hosdata[i].mode()[0], inplace = True )
        


# In[81]:


hosdata.isna().sum()


# ### Exploratory Data Analysis

# In[82]:


#Split columns into categorical and numerical columns

cat_cols = hosdata.select_dtypes(include = 'object').columns
cat_cols


# In[83]:


num_cols = hosdata.select_dtypes(include = ['int64', 'float64']).columns
num_cols


# ### Univariate Analysis
# 
# Plotting box plot to visualize the distribution and check for any outliers

# In[86]:


for i in num_cols:
    plt.figure(figsize = [8,5])
    print(i)
    sns.boxplot(hosdata[i])


# ##### Plotting pie charts to visualize the values distribution in each categorical column

# In[88]:


for i in cat_cols:
    print(hosdata[i].value_counts(normalize = True))
    plt.figure(figsize = [5,5])
    hosdata[i].value_counts(normalize = True).plot.pie(labeldistance = None, autopct = '%1.2f%%')
    plt.legend()
    print('-------------------------------------------------------------------------------------')
    


# >We can look percentage of values in category of columns and infer that, 'MSZoning', 'LandContour', 'Utilities', etc. columns are having more than 70% of a distribution in a single category

# ### Bivariate and Multivariate Analysis

# ##### MSZoning and LotFrontage

# In[90]:


sns.barplot(x = 'MSZoning', y = 'LotFrontage', data = hosdata)


# ##### MSSubclass and LotFrontage

# In[95]:


sns.barplot(x = 'MSSubClass', y = 'LotFrontage', data = hosdata);


# ##### HouseStyle vs SalePrice based on Street

# In[98]:


sns.barplot(x = 'HouseStyle', y = 'SalePrice', hue = 'Street', data = hosdata);


# ##### BldgType vs SalePrice

# In[100]:


sns.barplot(x = 'BldgType', y = 'SalePrice', data = hosdata);


# ##### BsmtQual vs SalePrice

# In[101]:


sns.barplot(x = 'BsmtQual', y = 'SalePrice', data = hosdata);


# #### Conclusions
# 
# - We can see that RL(Residential Low Density) has the highest lot frontage and RM(Residential Medium Density) has the least
# - We can see that 2 Story 1946  and newer has the highest lot frontage and PUD-MULTILEVEL-INCL-SPLIT-LEV/FOYER has the least
# - The SalePrice is not showing much variance with respect to the Style of dwelling(one story/two story)
# - The SalePrice is almost same for all the Building Types (Types of dwelling) and basement quality, so there is no significant pattern

# ##### Calculating Age of the  property
# 

# In[103]:



hosdata['Age'] = hosdata['YrSold'] - hosdata['YearBuilt']
hosdata['Age'].head()


# Now that we know the age of house, we don't really need the year sold and year built columns. So, we drop them.

# In[106]:


hosdata.drop(columns =  ['YearBuilt', 'YrSold'], axis = 1, inplace = True)


# In[107]:


hosdata.head()


# #### Correlationship between numerical columns

# In[112]:


plt.figure(figsize = [25,25])
sns.heatmap(hosdata.corr(numeric_only = True), annot = True, cmap = 'BuPu')
plt.title('correlation of numerical data')


# #### Get top 10 correlated columns

# In[116]:


k = 10
plt.figure(figsize = [15,15])
cols = hosdata.corr(numeric_only = True).nlargest(k, 'SalePrice').index
cm = np.corrcoef(hosdata[cols].values.T)
sns.heatmap(cm, annot = True, square = True, fmt = '.2f', cbar = True, annot_kws = {'size':10}, yticklabels = cols.values, xticklabels = cols.values)


# ##### We see that:
# 
# - GarageArea and GarageCars are highly correlated with coeff of 0.68
# - GrLivArea and TotRmsAbvGrd are highly correlated with coeff of 0.83
# - TotalBsmtSF and 1stFlrSF are highly correlated with coeff of 0.82

# #### PairPlot for Numerical Columns

# In[119]:


cols = ['OverallQual', 'SalePrice', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'Age']

plt.figure(figsize = [20,20])
sns.pairplot(hosdata[cols])


# ### Data Preparation

# ##### Dummy Encoding

# In[120]:


housing_num = hosdata.select_dtypes(include = ['int64', 'float64'])
housing_cat = hosdata.select_dtypes(include = 'object')


# In[122]:


housing_cat_dm = pd.get_dummies(housing_cat, drop_first = True, dtype = int)


# In[123]:


housing_cat_dm


# In[124]:


house = pd.concat([housing_num, housing_cat_dm], axis = 1)


# In[125]:


house.head()


# In[126]:


house.shape


# ##### Split into target and feature variables

# In[127]:


X = house.drop(['SalePrice'], axis = 1).copy()
y = house['SalePrice'].copy()


# In[128]:


X.head()


# In[129]:


y.head()


# In[133]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 


# In[134]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[140]:


X_train.head()


# In[141]:


print(X_train.shape)


# In[142]:


y_train.shape


# ##### Scaling the dataset with Standard Scaler

# In[143]:


num_cols = list(X_train.select_dtypes(include = ['int64', 'float64']).columns)


# In[144]:


scaler = StandardScaler()


# In[145]:


X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.fit_transform(X_test[num_cols])


# ##### Building a function to calculate evaluation matrics

# In[146]:


def eval_metrics(y_train, y_train_pred, y_test, y_pred):
    
    #r2 values for train and test data
    print('r2 score (train) = ', "%.2f" % r2_score(y_train, y_train_pred))
    print('r2 score (test) = ', '%.2f' % r2_score(y_test, y_pred))
    
    #RMSE for train and test data
    mse_train = mean_square_error(y_train, y_train_pred)
    mse_test = mean_square_error(y_test, y_pred)
    rmse_train = mse_train ** 0.5
    rmse_test = mse_test ** 0.5
    
    print('RMSE(Train) = ', "%.2f" % rmse_train)
    print('RMSE(Test) = ', "%.2f" % rmse_test)


# ### Model Building 

# In[150]:


import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


# In[151]:


X_train.shape


# ##### Top 15 feature selection by using RFE
# 

# In[156]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

n_features_to_select = 15  
rfe = RFE(linreg, n_features_to_select = 15)

rfe.fit(X_train, y_train)


# In[157]:


rfe.support_


# In[158]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[159]:


col = X_train.columns[rfe.support_]


# ### Model 1
# 
# ##### Assessing the model with Statsmodels

# In[161]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary() 


# In[162]:


selected_features = X_train.columns[rfe.support_]

linreg.fit(X_train[selected_features], y_train)

y_pred = linreg.predict(X_test[selected_features])


# In[163]:


y_pred[:10]


# 

# In[171]:





# In[ ]:




